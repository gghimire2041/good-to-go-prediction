#!/usr/bin/env python3
"""
API Testing Script for G2G Model

Tests the FastAPI endpoints with sample data.
"""

import json
import requests
import time
from typing import Dict, Any

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Sample test case
SAMPLE_CASE = {
    "gid": "test_case_001",
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
}

# High-risk case
HIGH_RISK_CASE = {
    "gid": "high_risk_001",
    "risk_category": "Critical",
    "region": "Asia Pacific",
    "business_type": "Energy",
    "regulatory_status": "Non-Compliant",
    "compliance_level": "Critical",
    "revenue": 1000000,
    "employee_count": 200,
    "years_in_business": 3,
    "credit_score": 400,
    "debt_to_equity_ratio": 3.5,
    "liquidity_ratio": 0.3,
    "market_share": 0.02,
    "growth_rate": -0.15,
    "profitability_margin": -0.05,
    "customer_satisfaction": 45,
    "operational_efficiency": 35,
    "regulatory_violations": 8,
    "audit_score": 25,
    "description": "A struggling energy company with multiple compliance issues and declining financial performance."
}


def test_health_check():
    """Test health check endpoint"""
    print("🔍 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False


def test_single_prediction(case: Dict[str, Any], case_name: str):
    """Test single prediction endpoint"""
    print(f"\n🔍 Testing single prediction for {case_name}...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=case,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful:")
            print(f"   GID: {data['gid']}")
            print(f"   G2G Score: {data['g2g_score']}")
            print(f"   Confidence: {data['confidence_level']}")
            return data
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None


def test_explanation(case: Dict[str, Any], case_name: str):
    """Test explanation endpoint"""
    print(f"\n🔍 Testing SHAP explanation for {case_name}...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/explain",
            json=case,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Explanation successful:")
            print(f"   GID: {data['gid']}")
            print(f"   G2G Score: {data['g2g_score']}")
            print(f"   Summary: {data['explanation_summary']}")
            
            print(f"\n   Top Positive Features:")
            for feature in data['top_positive_features'][:3]:
                print(f"     - {feature['feature']}: {feature['shap_value']:.4f}")
                
            print(f"\n   Top Negative Features:")
            for feature in data['top_negative_features'][:3]:
                print(f"     - {feature['feature']}: {feature['shap_value']:.4f}")
                
            return data
        else:
            print(f"❌ Explanation failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Explanation error: {e}")
        return None


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print(f"\n🔍 Testing batch prediction...")
    
    batch_request = {
        "cases": [SAMPLE_CASE, HIGH_RISK_CASE],
        "include_explanations": True
    }
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/batch_predict",
            json=batch_request,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch prediction successful:")
            print(f"   Total cases: {data['total_cases']}")
            print(f"   Processing time: {data['processing_time_seconds']}s")
            
            print(f"\n   Predictions:")
            for pred in data['predictions']:
                print(f"     - {pred['gid']}: {pred['g2g_score']} ({pred['confidence_level']})")
                
            if data['explanations']:
                print(f"\n   Explanations provided: {len(data['explanations'])}")
                
            return data
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        return None


def test_model_info():
    """Test model info endpoint"""
    print(f"\n🔍 Testing model info...")
    try:
        response = requests.get(f"{API_BASE_URL}/model/info")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved:")
            print(f"   Model type: {data['model_type']}")
            print(f"   Version: {data['version']}")
            print(f"   Features: {data['features']}")
            print(f"   Is fitted: {data['is_fitted']}")
            return data
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Model info error: {e}")
        return None


def performance_test():
    """Run performance test with multiple requests"""
    print(f"\n🚀 Running performance test...")
    
    num_requests = 10
    start_time = time.time()
    successful_requests = 0
    
    for i in range(num_requests):
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json=SAMPLE_CASE,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code == 200:
                successful_requests += 1
        except:
            pass
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"✅ Performance test completed:")
    print(f"   Total requests: {num_requests}")
    print(f"   Successful requests: {successful_requests}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average time per request: {total_time/num_requests:.3f}s")
    print(f"   Requests per second: {num_requests/total_time:.2f}")


def main():
    """Run all API tests"""
    print("=== G2G Model API Testing Suite ===")
    print(f"Testing API at: {API_BASE_URL}")
    
    # Wait a moment for the server to be ready
    print("\nWaiting for server to be ready...")
    time.sleep(2)
    
    # Health check
    if not test_health_check():
        print("❌ Server is not healthy. Stopping tests.")
        return
    
    # Test model info
    test_model_info()
    
    # Test single predictions
    test_single_prediction(SAMPLE_CASE, "Normal Case")
    test_single_prediction(HIGH_RISK_CASE, "High Risk Case")
    
    # Test explanations
    test_explanation(SAMPLE_CASE, "Normal Case")
    test_explanation(HIGH_RISK_CASE, "High Risk Case")
    
    # Test batch prediction
    test_batch_prediction()
    
    # Performance test
    performance_test()
    
    print("\n=== API Testing Complete ===")
    print("🎉 All tests completed successfully!")
    
    print("\n📊 API Documentation available at:")
    print(f"   Swagger UI: {API_BASE_URL}/docs")
    print(f"   ReDoc: {API_BASE_URL}/redoc")


if __name__ == "__main__":
    main()
