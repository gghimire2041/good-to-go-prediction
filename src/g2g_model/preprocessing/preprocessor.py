"""
G2G Model Preprocessing Pipeline

Handles text vectorization with TF-IDF, categorical feature encoding,
and numerical feature normalization for the G2G model.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class G2GPreprocessor:
    """
    Comprehensive preprocessing pipeline for G2G model data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the preprocessor
        
        Args:
            config: Configuration dictionary containing preprocessing parameters
        """
        self.config = config or self._get_default_config()
        
        # Initialize transformers
        self.text_vectorizer = None
        self.numerical_scaler = None
        self.label_encoders = {}
        self.feature_columns = {}
        self.is_fitted = False
        
        # Feature name mappings
        self.text_feature_names = []
        self.numerical_feature_names = []
        self.categorical_feature_names = []
        
        logger.info("G2G Preprocessor initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'text_features': ['description'],
            'categorical_features': ['risk_category', 'region', 'business_type', 
                                   'regulatory_status', 'compliance_level'],
            'numerical_features': ['revenue', 'employee_count', 'years_in_business',
                                 'credit_score', 'debt_to_equity_ratio', 'liquidity_ratio',
                                 'market_share', 'growth_rate', 'profitability_margin',
                                 'customer_satisfaction', 'operational_efficiency',
                                 'regulatory_violations', 'audit_score'],
            'target_feature': 'g2g_score',
            'id_feature': 'gid',
            'text_config': {
                'max_features': 1000,
                'min_df': 2,
                'max_df': 0.95,
                'ngram_range': (1, 2),
                'stop_words': 'english'
            },
            'numerical_config': {
                'scaler': 'standard',
                'handle_outliers': True,
                'outlier_method': 'iqr',
                'outlier_factor': 1.5
            }
        }
    
    def _handle_outliers(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Handle outliers in numerical columns using IQR method
        
        Args:
            df: Input DataFrame
            columns: List of columns to process
            
        Returns:
            DataFrame with outliers handled
        """
        df_clean = df.copy()
        outlier_info = {}
        
        for col in columns:
            if col not in df_clean.columns:
                continue
                
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            
            factor = self.config['numerical_config'].get('outlier_factor', 1.5)
            lower_bound = Q1 - factor * IQR
            upper_bound = Q3 + factor * IQR
            
            # Count outliers before handling
            outliers_count = ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            
            # Cap outliers
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
            
            outlier_info[col] = {
                'outliers_count': outliers_count,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        logger.info(f"Handled outliers for {len(columns)} numerical columns")
        return df_clean
    
    def _preprocess_text(self, texts: pd.Series) -> np.ndarray:
        """
        Preprocess text features using TF-IDF vectorization
        
        Args:
            texts: Series containing text data
            
        Returns:
            TF-IDF vectors as numpy array
        """
        if self.text_vectorizer is None:
            text_config = self.config['text_config']
            self.text_vectorizer = TfidfVectorizer(
                max_features=text_config.get('max_features', 1000),
                min_df=text_config.get('min_df', 2),
                max_df=text_config.get('max_df', 0.95),
                ngram_range=text_config.get('ngram_range', (1, 2)),
                stop_words=text_config.get('stop_words', 'english'),
                lowercase=True,
                strip_accents='unicode'
            )
            
            # Fit and transform
            vectors = self.text_vectorizer.fit_transform(texts.fillna(''))
            
            # Store feature names
            self.text_feature_names = [f"text_tfidf_{i}" for i in range(vectors.shape[1])]
            
        else:
            # Transform only
            vectors = self.text_vectorizer.transform(texts.fillna(''))
        
        return vectors.toarray()
    
    def _preprocess_categorical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Preprocess categorical features
        
        Args:
            df: Input DataFrame
            columns: List of categorical columns
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Categorical column {col} not found in data")
                continue
            
            if col not in self.label_encoders:
                # Fit new encoder
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df[col].fillna('Unknown'))
            else:
                # Transform using existing encoder
                # Handle unseen categories
                known_categories = set(self.label_encoders[col].classes_)
                df_temp = df[col].fillna('Unknown')
                
                # Replace unseen categories with 'Unknown'
                mask = ~df_temp.isin(known_categories)
                if mask.any():
                    logger.warning(f"Found {mask.sum()} unseen categories in {col}, replacing with 'Unknown'")
                    if 'Unknown' not in known_categories:
                        # Add 'Unknown' to classes if not already present
                        self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'Unknown')
                    df_temp.loc[mask] = 'Unknown'
                
                df_encoded[col] = self.label_encoders[col].transform(df_temp)
        
        return df_encoded
    
    def _preprocess_numerical(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Preprocess numerical features
        
        Args:
            df: Input DataFrame
            columns: List of numerical columns
            
        Returns:
            DataFrame with scaled numerical features
        """
        df_processed = df.copy()
        
        # Handle outliers if configured
        if self.config['numerical_config'].get('handle_outliers', True):
            df_processed = self._handle_outliers(df_processed, columns)
        
        # Scale features
        if self.numerical_scaler is None:
            scaler_type = self.config['numerical_config'].get('scaler', 'standard')
            
            if scaler_type == 'standard':
                self.numerical_scaler = StandardScaler()
            else:
                raise ValueError(f"Unsupported scaler type: {scaler_type}")
            
            # Fit and transform
            df_processed[columns] = self.numerical_scaler.fit_transform(df_processed[columns])
            
        else:
            # Transform only
            df_processed[columns] = self.numerical_scaler.transform(df_processed[columns])
        
        return df_processed
    
    def fit(self, df: pd.DataFrame) -> 'G2GPreprocessor':
        """
        Fit the preprocessor on training data
        
        Args:
            df: Training DataFrame
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting G2G preprocessor...")
        
        # Store feature column information
        self.feature_columns = {
            'text': self.config['text_features'],
            'categorical': self.config['categorical_features'], 
            'numerical': self.config['numerical_features'],
            'target': self.config['target_feature'],
            'id': self.config['id_feature']
        }
        
        # Validate required columns
        required_cols = (self.config['text_features'] + 
                        self.config['categorical_features'] + 
                        self.config['numerical_features'])
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Fit text vectorizer
        if self.config['text_features']:
            text_col = self.config['text_features'][0]  # Assuming single text column for now
            self._preprocess_text(df[text_col])
        
        # Fit categorical encoders
        if self.config['categorical_features']:
            self._preprocess_categorical(df, self.config['categorical_features'])
        
        # Fit numerical scaler
        if self.config['numerical_features']:
            self._preprocess_numerical(df, self.config['numerical_features'])
        
        self.is_fitted = True
        logger.info("G2G preprocessor fitted successfully")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Transform data using fitted preprocessor
        
        Args:
            df: Input DataFrame to transform
            
        Returns:
            Tuple of (transformed_features_array, feature_names_list)
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transforming data")
        
        logger.info("Transforming data with G2G preprocessor...")
        
        features = []
        feature_names = []
        
        # Process text features
        if self.config['text_features']:
            text_col = self.config['text_features'][0]
            text_features = self._preprocess_text(df[text_col])
            features.append(text_features)
            feature_names.extend(self.text_feature_names)
        
        # Process categorical features
        if self.config['categorical_features']:
            cat_df = self._preprocess_categorical(df, self.config['categorical_features'])
            cat_features = cat_df[self.config['categorical_features']].values
            features.append(cat_features)
            feature_names.extend(self.config['categorical_features'])
        
        # Process numerical features
        if self.config['numerical_features']:
            num_df = self._preprocess_numerical(df, self.config['numerical_features'])
            num_features = num_df[self.config['numerical_features']].values
            features.append(num_features)
            feature_names.extend(self.config['numerical_features'])
        
        # Concatenate all features
        if features:
            X = np.concatenate(features, axis=1)
        else:
            X = np.empty((len(df), 0))
        
        logger.info(f"Transformed data shape: {X.shape}")
        
        return X, feature_names
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Fit preprocessor and transform data in one step
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (transformed_features_array, feature_names_list)
        """
        return self.fit(df).transform(df)
    
    def get_target(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract target variable from DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            Target variable array
        """
        target_col = self.config['target_feature']
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in data")
        
        return df[target_col].values
    
    def save(self, filepath: str) -> None:
        """
        Save fitted preprocessor to file
        
        Args:
            filepath: Path to save the preprocessor
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted preprocessor")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for saving
        save_data = {
            'text_vectorizer': self.text_vectorizer,
            'numerical_scaler': self.numerical_scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'config': self.config,
            'text_feature_names': self.text_feature_names,
            'is_fitted': self.is_fitted
        }
        
        joblib.dump(save_data, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'G2GPreprocessor':
        """
        Load fitted preprocessor from file
        
        Args:
            filepath: Path to load the preprocessor from
            
        Returns:
            Loaded preprocessor instance
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
        
        save_data = joblib.load(filepath)
        
        # Create new instance
        preprocessor = cls(config=save_data['config'])
        
        # Restore fitted components
        preprocessor.text_vectorizer = save_data['text_vectorizer']
        preprocessor.numerical_scaler = save_data['numerical_scaler']
        preprocessor.label_encoders = save_data['label_encoders']
        preprocessor.feature_columns = save_data['feature_columns']
        preprocessor.text_feature_names = save_data['text_feature_names']
        preprocessor.is_fitted = save_data['is_fitted']
        
        # Derive full feature name list for downstream components
        try:
            preprocessor.feature_names = (
                (preprocessor.text_feature_names or [])
                + preprocessor.config.get('categorical_features', [])
                + preprocessor.config.get('numerical_features', [])
            )
        except Exception:
            # Fallback: leave unset; API will compute when loading
            preprocessor.feature_names = None  # type: ignore
        
        logger.info(f"Preprocessor loaded from {filepath}")
        
        return preprocessor
    
    def get_feature_importance_mapping(self) -> Dict[str, List[str]]:
        """
        Get mapping of feature types to their column names
        
        Returns:
            Dictionary mapping feature types to column names
        """
        mapping = {}
        
        if self.text_feature_names:
            mapping['text_features'] = self.text_feature_names
        if self.config['categorical_features']:
            mapping['categorical_features'] = self.config['categorical_features']
        if self.config['numerical_features']:
            mapping['numerical_features'] = self.config['numerical_features']
        
        return mapping


def create_preprocessor_from_config(config_path: str) -> G2GPreprocessor:
    """
    Create preprocessor from configuration file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured preprocessor instance
    """
    import yaml
    
    with open(config_path, 'r') as f:
        full_config = yaml.safe_load(f)
    
    # Extract preprocessing and features config
    preprocess_config = {
        **full_config['features'],
        'text_config': full_config['preprocessing']['text'],
        'numerical_config': full_config['preprocessing']['numerical']
    }
    
    return G2GPreprocessor(config=preprocess_config)


if __name__ == "__main__":
    print("Preprocessor is used via training and API components.")
