# G2G Model - Good-to-Go Scoring System

A professional-grade machine learning system for evaluating business readiness and compliance using CatBoost with SHAP explainability.

## 🚀 Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd g2g-modelling

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train and save the model
python scripts/train_and_save_model.py

# Start the API server
uvicorn src.g2g_model.api.main:app --host 0.0.0.0 --port 8000
```

## 📋 Overview

The G2G (Good-to-Go) model is a sophisticated machine learning system designed to evaluate business readiness and compliance across multiple dimensions. It combines financial health indicators, operational metrics, regulatory compliance data, and textual business descriptions to produce a comprehensive scoring system.

### Key Features

- **🤖 Advanced ML Pipeline**: CatBoost-based regression model with 241 engineered features
- **📊 Multi-Modal Input**: Handles numerical, categorical, and text features seamlessly  
- **🔍 SHAP Explainability**: Provides detailed explanations for every prediction
- **🚀 Production-Ready API**: FastAPI-based REST API with comprehensive validation
- **📈 Professional Architecture**: Modular, testable, and maintainable codebase
- **⚡ High Performance**: Optimized for both accuracy and inference speed

## 🏗️ Architecture

```
g2g-modelling/
├── src/g2g_model/
│   ├── data/               # Data generation and management
│   ├── preprocessing/      # Feature engineering and data preprocessing
│   ├── models/            # Model implementations
│   ├── evaluation/        # Model evaluation and SHAP explainability
│   ├── api/               # FastAPI application
│   └── config.py          # Configuration management
├── data/
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data and artifacts
├── models/                # Saved model artifacts
├── scripts/               # Training and utility scripts
├── tests/                 # Test suites
├── config/                # Configuration files
└── docs/                  # Documentation
```

## 🎯 Model Performance

The current model achieves:
- **R² Score**: 0.728 (72.8% variance explained)
- **RMSE**: 0.058 (highly accurate predictions)
- **Features**: 241 engineered features from 20 input variables
- **Training Time**: ~2 minutes on standard hardware

### Feature Importance (Top 10)

1. **debt_to_equity_ratio** (0.0267) - Financial leverage indicator
2. **liquidity_ratio** (0.0236) - Short-term financial health
3. **profitability_margin** (0.0189) - Business profitability
4. **regulatory_status** (0.0175) - Compliance standing
5. **growth_rate** (0.0165) - Business growth trajectory
6. **risk_category** (0.0155) - Overall risk assessment
7. **credit_score** (0.0145) - Creditworthiness indicator
8. **operational_efficiency** (0.0140) - Operational performance
9. **regulatory_violations** (0.0104) - Compliance history
10. **text_tfidf_87** (0.0098) - Text-derived insights

## 📊 Input Features

### Categorical Features (5)
- `risk_category`: Low, Medium, High, Critical
- `region`: Geographic region
- `business_type`: Industry classification
- `regulatory_status`: Current compliance status
- `compliance_level`: Overall compliance grade

### Numerical Features (13)
- `revenue`: Annual revenue
- `employee_count`: Number of employees
- `years_in_business`: Company age
- `credit_score`: Credit rating (300-850)
- `debt_to_equity_ratio`: Financial leverage
- `liquidity_ratio`: Short-term liquidity
- `market_share`: Market position (0-1)
- `growth_rate`: Revenue growth rate
- `profitability_margin`: Profit margins
- `customer_satisfaction`: Customer satisfaction score (0-100)
- `operational_efficiency`: Operational efficiency score (0-100)
- `regulatory_violations`: Number of violations
- `audit_score`: Latest audit score (0-100)

### Text Features (1)
- `description`: Business description (processed via TF-IDF)

## 🔧 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gid": "test_case_1",
    "risk_category": "Medium",
    "region": "North America",
    "business_type": "Technology",
    "regulatory_status": "Compliant",
    "compliance_level": "Good",
    "revenue": 5000000,
    "employee_count": 50,
    "years_in_business": 8,
    "credit_score": 720,
    "debt_to_equity_ratio": 0.5,
    "liquidity_ratio": 1.8,
    "market_share": 0.15,
    "growth_rate": 0.12,
    "profitability_margin": 0.15,
    "customer_satisfaction": 85,
    "operational_efficiency": 78,
    "regulatory_violations": 1,
    "audit_score": 82,
    "description": "A technology company specializing in cloud-based solutions with strong growth and good compliance record."
  }'
```

### Prediction with SHAP Explanation
```bash
curl -X POST "http://localhost:8000/explain" \
  -H "Content-Type: application/json" \
  -d '{
    # Same payload as above
  }'
```

### Batch Predictions
```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{
    "cases": [
      {
        # Multiple cases here
      }
    ],
    "include_explanations": true
  }'
```

## 🧪 Development

### Environment Setup
```bash
# Development dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Code formatting
black src/ scripts/ tests/
isort src/ scripts/ tests/

# Type checking
mypy src/

# Linting
flake8 src/ scripts/ tests/
```

### Training a New Model
```bash
# Generate new synthetic data and train model
python scripts/train_and_save_model.py

# The script will:
# 1. Generate 5000 synthetic samples
# 2. Preprocess the data
# 3. Train the CatBoost model
# 4. Evaluate with SHAP
# 5. Save model artifacts
```

### Model Evaluation
```python
from g2g_model.evaluation.evaluator import G2GModelEvaluator

# Create evaluator
evaluator = G2GModelEvaluator(model, feature_names)

# Run comprehensive evaluation
results = evaluator.evaluate_with_shap(X_train, y_train, X_test, y_test)

# Generate detailed report
report_path = evaluator.generate_model_report("evaluation_results/")
```

## 🐳 Deployment

### Docker Deployment
```bash
# Build Docker image
docker build -t g2g-model:latest .

# Run container
docker run -p 8000:8000 g2g-model:latest
```

### Production Considerations

1. **Monitoring**: Implement model drift detection
2. **Scaling**: Use multiple worker processes
3. **Security**: Add authentication and rate limiting
4. **Logging**: Comprehensive request/response logging
5. **Model Updates**: Implement A/B testing for model versions

## 📈 Model Interpretability

The system provides three levels of interpretability:

### 1. Global Feature Importance
- SHAP-based feature importance across all predictions
- Identifies which features drive model decisions most

### 2. Local Explanations  
- SHAP values for individual predictions
- Shows how each feature contributed to a specific score

### 3. Natural Language Explanations
- Human-readable summaries of predictions
- Highlights key positive and negative factors

Example explanation:
> "G2G Score: 0.742 (High confidence); Main positive factors: liquidity_ratio, profitability_margin; Main risk factors: debt_to_equity_ratio, regulatory_violations"

## 🔍 Model Validation

### Data Quality Checks
- Input validation using Pydantic models
- Range checking for numerical features
- Category validation for categorical features
- Text preprocessing and cleaning

### Model Performance
- Cross-validation during training
- Hold-out test set evaluation
- SHAP additivity checking
- Prediction consistency tests

### Monitoring
- Input drift detection
- Output distribution monitoring
- Performance degradation alerts
- Feature importance stability

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋‍♂️ Support

For questions, issues, or contributions:
- Create an issue in the repository
- Contact the development team
- Check the documentation in the `docs/` folder

---

**Built with ❤️ for professional ML deployments**
 
## 📚 Documentation

- Overview: docs/overview.md
- Architecture: docs/architecture.md
- API: docs/api.md
- Training: docs/training.md
- Testing: docs/testing.md
- Deployment: docs/deployment.md
- Operations: docs/operations.md
