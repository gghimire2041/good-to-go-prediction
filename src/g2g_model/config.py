"""
Configuration Management Module

Handles loading and validation of configuration settings for the G2G model.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseModel):
    """Data configuration settings"""
    raw_data_path: str
    processed_data_path: str
    train_test_split: float
    random_state: int
    n_samples: int

    @validator('train_test_split')
    def validate_split(cls, v):
        if not 0 < v < 1:
            raise ValueError('train_test_split must be between 0 and 1')
        return v


class FeaturesConfig(BaseModel):
    """Feature configuration settings"""
    text_features: List[str]
    categorical_features: List[str]
    numerical_features: List[str]
    target_feature: str
    id_feature: str


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration settings"""
    text: Dict[str, Any]
    numerical: Dict[str, Any]
    categorical: Dict[str, Any]


class ModelConfig(BaseModel):
    """Model configuration settings"""
    name: str
    hyperparameters: Dict[str, Any]
    cross_validation: Dict[str, Any]


class ExplainabilityConfig(BaseModel):
    """SHAP explainability configuration"""
    shap_explainer: str
    max_display: int
    save_explanations: bool
    explanation_path: str


class APIConfig(BaseModel):
    """API configuration settings"""
    host: str
    port: int
    title: str
    description: str
    version: str
    debug: bool


class LoggingConfig(BaseModel):
    """Logging configuration settings"""
    level: str
    log_file: str
    format: str


class ModelStorageConfig(BaseModel):
    """Model persistence configuration"""
    # Allow fields starting with "model_" without protected namespace warning
    model_config = ConfigDict(protected_namespaces=())
    model_path: str
    preprocessor_path: str
    feature_names_path: str
    model_metadata_path: str


class G2GConfig(BaseSettings):
    """Main configuration class"""
    # Keep only the default BaseSettings protected prefix and allow fields like model_storage
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        protected_namespaces=("settings_",),
    )
    data: DataConfig
    features: FeaturesConfig
    preprocessing: PreprocessingConfig
    model: ModelConfig
    explainability: ExplainabilityConfig
    api: APIConfig
    logging: LoggingConfig
    model_storage: ModelStorageConfig


def load_config(config_path: Optional[str] = None) -> G2GConfig:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file. If None, uses default path.
        
    Returns:
        G2GConfig: Loaded configuration object
    """
    if config_path is None:
        # Get the project root directory
        current_dir = Path(__file__).parent
        project_root = current_dir.parent.parent
        config_path = project_root / "config" / "config.yaml"
    
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return G2GConfig(**config_dict)


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent.parent


# Global configuration instance
config: Optional[G2GConfig] = None


def get_config() -> G2GConfig:
    """Get the global configuration instance"""
    global config
    if config is None:
        config = load_config()
    return config
