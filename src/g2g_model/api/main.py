"""
FastAPI Application for G2G Model

Production-ready REST API for G2G model inference with SHAP explanations.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from contextlib import asynccontextmanager
import json

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator, ConfigDict
import uvicorn
import numpy as np

# Add the source directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from g2g_model.models.catboost_model import G2GCatBoostModel
from g2g_model.preprocessing.preprocessor import G2GPreprocessor
from g2g_model.evaluation.evaluator import G2GModelEvaluator
from g2g_model.config import get_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables to store model and preprocessor
model: Optional[G2GCatBoostModel] = None
preprocessor: Optional[G2GPreprocessor] = None
evaluator: Optional[G2GModelEvaluator] = None
feature_names: Optional[List[str]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events for the FastAPI application"""
    # Startup
    try:
        log_artifact_validation_summary()
    except Exception as e:
        logger.warning(f"Artifact validation summary failed: {e}")
    await load_model()
    yield
    # Shutdown - cleanup if needed
    pass


# Pydantic models for request/response
class G2GInput(BaseModel):
    """Input schema for G2G prediction"""
    gid: Optional[str] = Field(None, description="Unique identifier for the case")
    
    # Categorical features
    risk_category: str = Field(..., description="Risk category (Low, Medium, High, Critical)")
    region: str = Field(..., description="Geographic region")
    business_type: str = Field(..., description="Type of business")
    regulatory_status: str = Field(..., description="Current regulatory status")
    compliance_level: str = Field(..., description="Compliance level")
    
    # Numerical features
    revenue: float = Field(..., ge=0, description="Annual revenue")
    employee_count: int = Field(..., ge=1, description="Number of employees")
    years_in_business: int = Field(..., ge=1, description="Years in business")
    credit_score: int = Field(..., ge=300, le=850, description="Credit score")
    debt_to_equity_ratio: float = Field(..., ge=0, description="Debt to equity ratio")
    liquidity_ratio: float = Field(..., ge=0, description="Liquidity ratio")
    market_share: float = Field(..., ge=0, le=1, description="Market share (0-1)")
    growth_rate: float = Field(..., description="Growth rate")
    profitability_margin: float = Field(..., description="Profitability margin")
    customer_satisfaction: float = Field(..., ge=0, le=100, description="Customer satisfaction score")
    operational_efficiency: float = Field(..., ge=0, le=100, description="Operational efficiency score")
    regulatory_violations: int = Field(..., ge=0, description="Number of regulatory violations")
    audit_score: float = Field(..., ge=0, le=100, description="Audit score")
    
    # Text feature
    description: str = Field(..., description="Business description")
    
    @validator('risk_category')
    def validate_risk_category(cls, v):
        allowed = ['Low', 'Medium', 'High', 'Critical']
        if v not in allowed:
            raise ValueError(f'risk_category must be one of {allowed}')
        return v
    
    @validator('regulatory_status')
    def validate_regulatory_status(cls, v):
        allowed = ['Compliant', 'Under Review', 'Minor Issues', 'Major Issues', 'Non-Compliant']
        if v not in allowed:
            raise ValueError(f'regulatory_status must be one of {allowed}')
        return v
    
    @validator('compliance_level')
    def validate_compliance_level(cls, v):
        allowed = ['Excellent', 'Good', 'Fair', 'Poor', 'Critical']
        if v not in allowed:
            raise ValueError(f'compliance_level must be one of {allowed}')
        return v


class G2GPrediction(BaseModel):
    """Response schema for G2G prediction"""
    # Allow fields starting with "model_"
    model_config = ConfigDict(protected_namespaces=())
    gid: str
    g2g_score: float = Field(..., description="Predicted G2G score (0-1)")
    confidence_level: str = Field(..., description="High/Medium/Low confidence")
    prediction_timestamp: str
    model_version: str


class G2GExplanation(BaseModel):
    """Response schema for SHAP explanation"""
    model_config = ConfigDict(protected_namespaces=())
    gid: str
    g2g_score: float
    explanation_summary: str
    top_positive_features: List[Dict[str, Any]]
    top_negative_features: List[Dict[str, Any]]
    feature_contributions: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    """Health check response"""
    model_config = ConfigDict(protected_namespaces=())
    status: str
    model_loaded: bool
    model_version: str
    uptime: str


class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    cases: List[G2GInput]
    include_explanations: bool = Field(default=False, description="Include SHAP explanations")


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    predictions: List[G2GPrediction]
    explanations: Optional[List[G2GExplanation]] = None
    total_cases: int
    processing_time_seconds: float


# FastAPI app initialization
app = FastAPI(
    title="G2G Model API",
    description="Good-to-Go Model Inference API with SHAP Explanations",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def load_model():
    """Load the trained model, preprocessor, and evaluator"""
    global model, preprocessor, evaluator, feature_names
    
    try:
        logger.info("Loading G2G model and components...")
        
        # Try to load saved model first
        model_path = Path("models/g2g_model.pkl")
        preprocessor_path = Path("models/preprocessor.pkl")
        
        if model_path.exists() and preprocessor_path.exists():
            logger.info("Loading saved model and preprocessor...")
            try:
                model = G2GCatBoostModel.load(str(model_path))
                preprocessor = G2GPreprocessor.load(str(preprocessor_path))
                # Compute feature names deterministically from the preprocessor
                if getattr(preprocessor, 'feature_names', None):
                    feature_names = preprocessor.feature_names  # type: ignore
                else:
                    text_names = getattr(preprocessor, 'text_feature_names', []) or []
                    feature_names = text_names + \
                        preprocessor.config.get('categorical_features', []) + \
                        preprocessor.config.get('numerical_features', [])
                logger.info("Model and preprocessor loaded successfully")
            except Exception as e:
                logger.warning(
                    "Failed to load saved artifacts (version mismatch or missing files). "
                    f"Fallback to training demo model. Details: {e}"
                )
                await train_demo_model()
        else:
            # Train a new model for demo purposes
            logger.warning("Saved model not found. Training new model for demo...")
            await train_demo_model()
        
        # Initialize evaluator
        evaluator = G2GModelEvaluator(model, feature_names)
        logger.info("G2G Model API components loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def log_artifact_validation_summary() -> None:
    """Log a concise validation summary for artifacts if present."""
    model_path = Path("models/g2g_model.pkl")
    preprocessor_path = Path("models/preprocessor.pkl")
    metadata_path = model_path.with_suffix('.json')

    if not (model_path.exists() and preprocessor_path.exists()):
        logger.info("Artifacts not found; API may train a demo model on startup.")
        return

    model_ok = False
    pre_ok = False
    feat_meta = None
    feat_pre = None
    versions = {}

    # Load metadata if available
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
            feat_meta = meta.get('feature_names')
            versions = meta.get('versions', {})
        except Exception as e:
            logger.warning(f"Failed to read model metadata: {e}")

    # Try load preprocessor and derive features
    try:
        pre = G2GPreprocessor.load(str(preprocessor_path))
        if getattr(pre, 'feature_names', None):
            feat_pre = pre.feature_names  # type: ignore
        else:
            text_names = getattr(pre, 'text_feature_names', []) or []
            feat_pre = text_names + pre.config.get('categorical_features', []) + pre.config.get('numerical_features', [])
        pre_ok = True
    except Exception as e:
        logger.warning(f"Preprocessor load failed: {e}")

    # Try load model
    try:
        _ = G2GCatBoostModel.load(str(model_path))
        model_ok = True
    except Exception as e:
        logger.warning(f"Model load failed: {e}")

    # Compose summary
    feat_meta_n = len(feat_meta) if isinstance(feat_meta, list) else None
    feat_pre_n = len(feat_pre) if isinstance(feat_pre, list) else None

    msg = (
        f"Artifacts summary — model: {'ok' if model_ok else 'fail'}, "
        f"preprocessor: {'ok' if pre_ok else 'fail'}, "
        f"features(meta/pre): {feat_meta_n}/{feat_pre_n}, "
        f"versions: numpy={versions.get('numpy','?')}, sklearn={versions.get('sklearn','?')}, catboost={versions.get('catboost','?')}"
    )
    logger.info(msg)


async def train_demo_model():
    """Train a demo model if no saved model exists"""
    global model, preprocessor, feature_names
    
    try:
        # Import required modules
        from g2g_model.data.data_generator import G2GDataGenerator
        from sklearn.model_selection import train_test_split
        
        logger.info("Generating demo data and training model...")
        
        # Generate synthetic data
        generator = G2GDataGenerator(random_state=42)
        df = generator.generate_data(n_samples=1000)  # Smaller for demo
        
        # Preprocess data
        preprocessor = G2GPreprocessor()
        X, feature_names = preprocessor.fit_transform(df)
        y = preprocessor.get_target(df)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        config = {
            'hyperparameters': {
                'iterations': 100,
                'learning_rate': 0.1,
                'depth': 6,
                'verbose': False
            }
        }
        
        model = G2GCatBoostModel(config=config)
        model.fit(X_train, y_train, feature_names)
        
        logger.info("Demo model trained successfully")
        
    except Exception as e:
        logger.error(f"Failed to train demo model: {e}")
        raise


def convert_input_to_dataframe(g2g_input: G2GInput) -> pd.DataFrame:
    """Convert G2GInput to DataFrame for preprocessing"""
    data = {
        'gid': g2g_input.gid or 'api_request',
        'risk_category': g2g_input.risk_category,
        'region': g2g_input.region,
        'business_type': g2g_input.business_type,
        'regulatory_status': g2g_input.regulatory_status,
        'compliance_level': g2g_input.compliance_level,
        'revenue': g2g_input.revenue,
        'employee_count': g2g_input.employee_count,
        'years_in_business': g2g_input.years_in_business,
        'credit_score': g2g_input.credit_score,
        'debt_to_equity_ratio': g2g_input.debt_to_equity_ratio,
        'liquidity_ratio': g2g_input.liquidity_ratio,
        'market_share': g2g_input.market_share,
        'growth_rate': g2g_input.growth_rate,
        'profitability_margin': g2g_input.profitability_margin,
        'customer_satisfaction': g2g_input.customer_satisfaction,
        'operational_efficiency': g2g_input.operational_efficiency,
        'regulatory_violations': g2g_input.regulatory_violations,
        'audit_score': g2g_input.audit_score,
        'description': g2g_input.description
    }
    
    return pd.DataFrame([data])


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "G2G Model API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    import time
    
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        model_version="1.0.0",
        uptime=str(time.time())
    )


@app.post("/predict", response_model=G2GPrediction)
async def predict(g2g_input: G2GInput):
    """
    Predict G2G score for a single case
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        df = convert_input_to_dataframe(g2g_input)
        
        # Preprocess
        X, _ = preprocessor.transform(df)
        
        # Predict
        prediction = model.predict(X)[0]
        
        # Determine confidence level
        if prediction > 0.7:
            confidence = "High"
        elif prediction > 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        import datetime
        
        return G2GPrediction(
            gid=g2g_input.gid or "api_request",
            g2g_score=round(float(prediction), 4),
            confidence_level=confidence,
            prediction_timestamp=datetime.datetime.now().isoformat(),
            model_version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/explain", response_model=G2GExplanation)
async def explain_prediction(g2g_input: G2GInput):
    """
    Predict G2G score with SHAP explanation
    """
    if model is None or preprocessor is None or evaluator is None:
        raise HTTPException(status_code=503, detail="Model or explainer not loaded")
    
    try:
        # Convert input to DataFrame
        df = convert_input_to_dataframe(g2g_input)
        
        # Preprocess
        X, _ = preprocessor.transform(df)
        
        # Initialize SHAP explainer if not already done
        if evaluator.explainer is None:
            # Use a small background dataset for initialization
            background_data = np.random.randn(10, X.shape[1])  # Demo background
            evaluator._initialize_shap_explainer(background_data)
        
        # Get explanation
        explanation = evaluator.explain_prediction(X[0], g2g_input.gid or "api_request")
        
        return G2GExplanation(
            gid=explanation['instance_id'],
            g2g_score=round(float(explanation['prediction']), 4),
            explanation_summary=explanation['explanation_summary'],
            top_positive_features=explanation['top_positive_features'],
            top_negative_features=explanation['top_negative_features'],
            feature_contributions=explanation['feature_contributions']
        )
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """
    Batch prediction for multiple cases
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    import time
    start_time = time.time()
    
    try:
        predictions = []
        explanations = [] if request.include_explanations else None
        
        for case in request.cases:
            # Convert to DataFrame and preprocess
            df = convert_input_to_dataframe(case)
            X, _ = preprocessor.transform(df)
            
            # Predict
            prediction = model.predict(X)[0]
            confidence = "High" if prediction > 0.7 else "Medium" if prediction > 0.4 else "Low"
            
            predictions.append(G2GPrediction(
                gid=case.gid or f"batch_{len(predictions)}",
                g2g_score=round(float(prediction), 4),
                confidence_level=confidence,
                prediction_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
                model_version="1.0.0"
            ))
            
            # Add explanation if requested
            if request.include_explanations and evaluator is not None:
                try:
                    if evaluator.explainer is None:
                        background_data = np.random.randn(10, X.shape[1])
                        evaluator._initialize_shap_explainer(background_data)
                    
                    explanation = evaluator.explain_prediction(X[0], case.gid or f"batch_{len(predictions)-1}")
                    explanations.append(G2GExplanation(
                        gid=explanation['instance_id'],
                        g2g_score=round(float(explanation['prediction']), 4),
                        explanation_summary=explanation['explanation_summary'],
                        top_positive_features=explanation['top_positive_features'],
                        top_negative_features=explanation['top_negative_features'],
                        feature_contributions=explanation['feature_contributions']
                    ))
                except Exception as e:
                    logger.warning(f"Failed to generate explanation for case: {e}")
        
        processing_time = time.time() - start_time
        
        return BatchPredictionResponse(
            predictions=predictions,
            explanations=explanations,
            total_cases=len(request.cases),
            processing_time_seconds=round(processing_time, 3)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/model/info")
async def get_model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        info = {
            "model_type": "CatBoost",
            "version": "1.0.0",
            "features": len(feature_names) if feature_names else 0,
            "feature_names": feature_names[:20] if feature_names else [],  # First 20 features
            "training_config": model.config if hasattr(model, 'config') else {},
            "is_fitted": model.is_fitted if hasattr(model, 'is_fitted') else False
        }
        
        if hasattr(model, 'training_history') and model.training_history:
            info["training_history"] = model.training_history
        
        return info
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
