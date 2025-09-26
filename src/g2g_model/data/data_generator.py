"""
Synthetic Data Generator for G2G Model

Generates realistic synthetic data with 20 high-potential features including
text, categorical, and numerical features with realistic correlations to
G2G scores.
"""

import pandas as pd
import numpy as np
import uuid
from typing import Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class G2GDataGenerator:
    """
    Generates synthetic G2G (Good-to-Go) data with realistic business features
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the data generator
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
        # Define realistic categorical values
        self.risk_categories = ['Low', 'Medium', 'High', 'Critical']
        self.regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East', 'Africa']
        self.business_types = [
            'Manufacturing', 'Technology', 'Healthcare', 'Finance', 'Retail', 'Energy',
            'Telecommunications'
        ]
        self.regulatory_statuses = ['Compliant', 'Under Review', 'Minor Issues', 'Major Issues', 'Non-Compliant']
        self.compliance_levels = ['Excellent', 'Good', 'Fair', 'Poor', 'Critical']
        
        # Business description templates for text generation
        self.description_templates = [
            "A {} company operating in {} with {} track record in regulatory compliance",
            "Leading {} organization based in {} specializing in innovative solutions",
            "Established {} business in {} market with strong financial performance",
            "Growing {} enterprise focused on {} operations and customer satisfaction",
            "International {} company with {} presence and diversified portfolio",
            "Regional {} leader in {} known for operational excellence",
            "Emerging {} startup in {} with disruptive business model",
            "Traditional {} firm in {} with conservative growth strategy"
        ]
        
        self.compliance_descriptors = ['strong', 'excellent', 'good', 'adequate', 'weak', 'poor', 'concerning']
        self.performance_descriptors = ['outstanding', 'solid', 'stable', 'moderate', 'declining', 'volatile']
    
    def _generate_text_feature(self, row: Dict[str, Any]) -> str:
        """
        Generate realistic business descriptions based on other features
        
        Args:
            row: Dictionary containing other feature values
            
        Returns:
            Generated text description
        """
        # template = np.random.choice(self.description_templates)
        business_type = row['business_type'].lower()
        region = row['region'].lower()
        
        # Add performance indicators based on numerical features
        if row['profitability_margin'] > 0.15:
            performance = np.random.choice(['outstanding', 'excellent', 'strong'])
        elif row['profitability_margin'] > 0.08:
            performance = np.random.choice(['solid', 'good', 'stable'])
        else:
            performance = np.random.choice(['moderate', 'declining', 'challenging'])
        
        # Add compliance indicators
        compliance_map = {
            'Excellent': 'excellent', 'Good': 'strong', 'Fair': 'adequate',
            'Poor': 'weak', 'Critical': 'concerning'
        }
        compliance_desc = compliance_map.get(row['compliance_level'], 'standard')
        
        # Create contextual description
        descriptions = [
            (
                f"A {business_type} company operating in {region} with {compliance_desc} "
                f"regulatory compliance and {performance} financial performance."
            ),
            (
                f"Leading {business_type} organization based in {region} showing {performance} "
                f"growth trajectory and maintaining {compliance_desc} compliance standards."
            ),
            (
                f"Established {business_type} business in {region} market with {performance} "
                f"operational metrics and {compliance_desc} regulatory standing."
            ),
        ]
        
        return np.random.choice(descriptions)
    
    def _calculate_g2g_score(self, row: Dict[str, Any]) -> float:
        """
        Calculate G2G score based on feature values with realistic correlations
        
        Args:
            row: Dictionary containing feature values
            
        Returns:
            G2G score between 0 and 1
        """
        # Base score calculation with weighted features
        score = 0.0
        
        # Financial health (40% weight)
        financial_score = (
            (row['credit_score'] / 850) * 0.3 +
            (1 - row['debt_to_equity_ratio'] / 3) * 0.25 +
            (row['liquidity_ratio'] / 3) * 0.2 +
            (row['profitability_margin'] / 0.3) * 0.25
        )
        score += financial_score * 0.4
        
        # Operational efficiency (25% weight)
        operational_score = (
            (row['operational_efficiency'] / 100) * 0.4 +
            (row['customer_satisfaction'] / 100) * 0.3 +
            (row['growth_rate'] / 0.5) * 0.3
        )
        score += operational_score * 0.25
        
        # Compliance and risk (25% weight)
        compliance_map = {'Excellent': 1.0, 'Good': 0.8, 'Fair': 0.6, 'Poor': 0.3, 'Critical': 0.1}
        risk_map = {'Low': 1.0, 'Medium': 0.7, 'High': 0.4, 'Critical': 0.1}
        regulatory_map = {
            'Compliant': 1.0,
            'Under Review': 0.7,
            'Minor Issues': 0.5,
            'Major Issues': 0.2,
            'Non-Compliant': 0.0,
        }
        
        compliance_score = (
            compliance_map[row['compliance_level']] * 0.4 +
            risk_map[row['risk_category']] * 0.3 +
            regulatory_map[row['regulatory_status']] * 0.2 +
            (1 - row['regulatory_violations'] / 10) * 0.1
        )
        score += compliance_score * 0.25
        
        # Market position (10% weight)
        market_score = (
            (row['market_share'] / 0.5) * 0.4 +
            (row['years_in_business'] / 50) * 0.3 +
            (row['audit_score'] / 100) * 0.3
        )
        score += market_score * 0.1
        
        # Add some noise and ensure score is between 0 and 1
        noise = np.random.normal(0, 0.05)
        score = max(0.0, min(1.0, score + noise))
        
        return round(score, 4)
    
    def generate_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """
        Generate synthetic G2G dataset
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic G2G data
        """
        logger.info(f"Generating {n_samples} synthetic G2G samples...")
        
        data = []
        
        for i in range(n_samples):
            # Generate unique ID
            gid = str(uuid.uuid4())
            
            # Generate categorical features
            risk_category = np.random.choice(self.risk_categories, p=[0.3, 0.4, 0.25, 0.05])
            region = np.random.choice(self.regions)
            business_type = np.random.choice(self.business_types)
            
            # Regulatory status correlates with risk category
            if risk_category == 'Low':
                regulatory_status = np.random.choice(self.regulatory_statuses, p=[0.7, 0.2, 0.08, 0.02, 0.0])
            elif risk_category == 'Medium':
                regulatory_status = np.random.choice(self.regulatory_statuses, p=[0.4, 0.3, 0.2, 0.08, 0.02])
            elif risk_category == 'High':
                regulatory_status = np.random.choice(self.regulatory_statuses, p=[0.1, 0.2, 0.3, 0.3, 0.1])
            else:  # Critical
                regulatory_status = np.random.choice(self.regulatory_statuses, p=[0.0, 0.1, 0.2, 0.4, 0.3])
            
            # Compliance level correlates with regulatory status
            compliance_map = {
                'Compliant': np.random.choice(self.compliance_levels, p=[0.6, 0.3, 0.08, 0.02, 0.0]),
                'Under Review': np.random.choice(self.compliance_levels, p=[0.2, 0.4, 0.3, 0.08, 0.02]),
                'Minor Issues': np.random.choice(self.compliance_levels, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
                'Major Issues': np.random.choice(self.compliance_levels, p=[0.02, 0.08, 0.3, 0.4, 0.2]),
                'Non-Compliant': np.random.choice(self.compliance_levels, p=[0.0, 0.05, 0.15, 0.3, 0.5])
            }
            compliance_level = compliance_map[regulatory_status]
            
            # Generate numerical features with realistic correlations
            # Financial features
            revenue = np.random.lognormal(15, 2)  # Log-normal distribution for revenue
            employee_count = max(1, int(np.random.lognormal(4, 1.5)))
            years_in_business = max(1, int(np.random.exponential(8)))
            
            # Credit score correlates with risk category
            if risk_category == 'Low':
                credit_score = int(np.random.normal(750, 50))
            elif risk_category == 'Medium':
                credit_score = int(np.random.normal(650, 60))
            elif risk_category == 'High':
                credit_score = int(np.random.normal(550, 70))
            else:  # Critical
                credit_score = int(np.random.normal(450, 80))
            
            credit_score = max(300, min(850, credit_score))
            
            # Financial ratios
            debt_to_equity_ratio = max(0, np.random.lognormal(0, 0.8))
            liquidity_ratio = max(0.1, np.random.lognormal(0.5, 0.6))
            
            # Market and operational metrics
            market_share = max(0, min(0.5, np.random.beta(2, 8)))
            growth_rate = np.random.normal(0.08, 0.15)
            profitability_margin = np.random.normal(0.1, 0.08)
            
            customer_satisfaction = max(0, min(100, np.random.normal(75, 15)))
            operational_efficiency = max(0, min(100, np.random.normal(70, 20)))
            
            # Compliance-related metrics
            if compliance_level in ['Excellent', 'Good']:
                regulatory_violations = max(0, int(np.random.poisson(0.5)))
                audit_score = max(0, min(100, np.random.normal(85, 10)))
            elif compliance_level == 'Fair':
                regulatory_violations = max(0, int(np.random.poisson(2)))
                audit_score = max(0, min(100, np.random.normal(70, 15)))
            else:  # Poor or Critical
                regulatory_violations = max(0, int(np.random.poisson(5)))
                audit_score = max(0, min(100, np.random.normal(50, 20)))
            
            # Create row dictionary
            row = {
                'gid': gid,
                'risk_category': risk_category,
                'region': region,
                'business_type': business_type,
                'regulatory_status': regulatory_status,
                'compliance_level': compliance_level,
                'revenue': revenue,
                'employee_count': employee_count,
                'years_in_business': years_in_business,
                'credit_score': credit_score,
                'debt_to_equity_ratio': debt_to_equity_ratio,
                'liquidity_ratio': liquidity_ratio,
                'market_share': market_share,
                'growth_rate': growth_rate,
                'profitability_margin': profitability_margin,
                'customer_satisfaction': customer_satisfaction,
                'operational_efficiency': operational_efficiency,
                'regulatory_violations': regulatory_violations,
                'audit_score': audit_score
            }
            
            # Generate text description
            row['description'] = self._generate_text_feature(row)
            
            # Calculate G2G score
            row['g2g_score'] = self._calculate_g2g_score(row)
            
            data.append(row)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Generated {i + 1} samples...")
        
        df = pd.DataFrame(data)
        logger.info(f"Successfully generated {len(df)} samples with {len(df.columns)} features")
        
        return df
    
    def save_data(self, df: pd.DataFrame, filepath: str) -> None:
        """
        Save generated data to CSV file
        
        Args:
            df: DataFrame to save
            filepath: Path to save the file
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(filepath, index=False)
        logger.info(f"Data saved to {filepath}")
    
    def get_feature_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics of generated features
        
        Args:
            df: Generated DataFrame
            
        Returns:
            Dictionary containing feature summaries
        """
        summary = {
            'total_samples': len(df),
            'target_distribution': {
                'mean': df['g2g_score'].mean(),
                'std': df['g2g_score'].std(),
                'min': df['g2g_score'].min(),
                'max': df['g2g_score'].max(),
                'quartiles': df['g2g_score'].quantile([0.25, 0.5, 0.75]).to_dict()
            },
            'categorical_features': {},
            'numerical_features': {}
        }
        
        # Categorical feature distributions
        categorical_cols = ['risk_category', 'region', 'business_type', 'regulatory_status', 'compliance_level']
        for col in categorical_cols:
            summary['categorical_features'][col] = df[col].value_counts().to_dict()
        
        # Numerical feature statistics
        numerical_cols = ['revenue', 'employee_count', 'years_in_business', 'credit_score', 
                         'debt_to_equity_ratio', 'liquidity_ratio', 'market_share', 'growth_rate',
                         'profitability_margin', 'customer_satisfaction', 'operational_efficiency',
                         'regulatory_violations', 'audit_score']
        
        for col in numerical_cols:
            summary['numerical_features'][col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max()
            }
        
        return summary


if __name__ == "__main__":
    # Example usage
    generator = G2GDataGenerator(random_state=42)
    data = generator.generate_data(n_samples=5000)
    generator.save_data(data, "data/raw/g2g_dataset.csv")
    
    # Print summary
    summary = generator.get_feature_summary(data)
    print(f"Generated {summary['total_samples']} samples")
    mean = summary['target_distribution']['mean']
    std = summary['target_distribution']['std']
    print(f"G2G Score - Mean: {mean:.3f}, Std: {std:.3f}")
