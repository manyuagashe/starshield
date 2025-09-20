#!/usr/bin/env python3
"""
Test script for the StarShield Asteroid Risk Prediction API
"""
import requests
import json
import time

# Wait for server to start
print("🚀 Testing StarShield API...")
time.sleep(3)

base_url = "http://localhost:8000"

def test_health():
    """Test health endpoints"""
    print("\n📊 Testing Health Endpoints...")
    
    try:
        response = requests.get(f"{base_url}/health/live")
        print(f"   Health Live: {response.status_code} - {response.json()}")
        
        response = requests.get(f"{base_url}/")
        print(f"   Root Health: {response.status_code} - {response.json()}")
        
        return True
    except Exception as e:
        print(f"   ❌ Health test failed: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\n🧠 Testing Model Info...")
    
    try:
        response = requests.get(f"{base_url}/model/info")
        info = response.json()
        print(f"   Model Info: {response.status_code}")
        print(f"   Model Type: {info.get('model_type')}")
        print(f"   Training Records: {info.get('training_records')}")
        print(f"   Features: {info.get('features_used')}")
        print(f"   Classes: {info.get('available_classes')}")
        return True
    except Exception as e:
        print(f"   ❌ Model info test failed: {e}")
        return False

def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n🎯 Testing Single Prediction...")
    
    test_asteroid = {
        "distance_au": 0.015,
        "velocity_kms": 8.5,
        "diameter_km": 0.025,
        "v_infinity_kms": 12.3,
        "is_pha": False,
        "orbit_class": "AMO"
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=test_asteroid)
        prediction = response.json()
        print(f"   Single Prediction: {response.status_code}")
        print(f"   Risk Level: {prediction.get('predicted_risk_level')}")
        print(f"   Confidence: {prediction.get('confidence')}")
        print(f"   Processing Time: {prediction.get('processing_time_ms')}ms")
        return True
    except Exception as e:
        print(f"   ❌ Single prediction test failed: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n📦 Testing Batch Prediction...")
    
    batch_data = {
        "asteroids": [
            {
                "distance_au": 0.015,
                "velocity_kms": 8.5,
                "diameter_km": 0.025,
                "v_infinity_kms": 12.3,
                "is_pha": False,
                "orbit_class": "AMO"
            },
            {
                "distance_au": 0.008,
                "velocity_kms": 15.2,
                "diameter_km": 0.045,
                "v_infinity_kms": 18.7,
                "is_pha": True,
                "orbit_class": "ATE"
            }
        ]
    }
    
    try:
        response = requests.post(f"{base_url}/predict/batch", json=batch_data)
        batch_result = response.json()
        print(f"   Batch Prediction: {response.status_code}")
        print(f"   Total Predictions: {batch_result.get('total_predictions')}")
        print(f"   Successful: {batch_result.get('successful_predictions')}")
        print(f"   Failed: {batch_result.get('failed_predictions')}")
        print(f"   Processing Time: {batch_result.get('total_processing_time_ms')}ms")
        
        if batch_result.get('predictions'):
            for i, pred in enumerate(batch_result['predictions'][:2]):  # Show first 2
                print(f"     Asteroid {i+1}: {pred.get('predicted_risk_level')} (confidence: {pred.get('confidence')})")
        return True
    except Exception as e:
        print(f"   ❌ Batch prediction test failed: {e}")
        return False

def test_train_data_endpoint():
    """Test GET endpoint for training data predictions"""
    print("\n🏋️ Testing Training Data Endpoint...")
    
    try:
        response = requests.get(f"{base_url}/data/train")
        train_result = response.json()
        print(f"   Train Data: {response.status_code}")
        print(f"   Total Predictions: {train_result.get('total_predictions')}")
        print(f"   Successful: {train_result.get('successful_predictions')}")
        print(f"   Failed: {train_result.get('failed_predictions')}")
        print(f"   Processing Time: {train_result.get('total_processing_time_ms')}ms")
        
        if train_result.get('predictions'):
            # Show risk distribution
            risk_counts = {}
            for pred in train_result['predictions']:
                risk = pred.get('predicted_risk_level')
                risk_counts[risk] = risk_counts.get(risk, 0) + 1
            
            print(f"   Risk Distribution:")
            for risk, count in risk_counts.items():
                percentage = (count / len(train_result['predictions'])) * 100
                print(f"     {risk}: {count} ({percentage:.1f}%)")
        
        return True
    except Exception as e:
        print(f"   ❌ Training data test failed: {e}")
        return False

def test_test_data_endpoint():
    """Test GET endpoint for test data predictions"""
    print("\n🧪 Testing Test Data Endpoint...")
    
    try:
        response = requests.get(f"{base_url}/data/test")
        test_result = response.json()
        print(f"   Test Data: {response.status_code}")
        print(f"   Total Predictions: {test_result.get('total_predictions')}")
        print(f"   Successful: {test_result.get('successful_predictions')}")
        print(f"   Failed: {test_result.get('failed_predictions')}")
        print(f"   Processing Time: {test_result.get('total_processing_time_ms')}ms")
        
        if test_result.get('predictions'):
            # Show risk distribution
            risk_counts = {}
            for pred in test_result['predictions']:
                risk = pred.get('predicted_risk_level')
                risk_counts[risk] = risk_counts.get(risk, 0) + 1
            
            print(f"   Risk Distribution:")
            for risk, count in risk_counts.items():
                percentage = (count / len(test_result['predictions'])) * 100
                print(f"     {risk}: {count} ({percentage:.1f}%)")
        
        return test_result
    except Exception as e:
        print(f"   ❌ Test data test failed: {e}")
        return False

def evaluate_accuracy(test_predictions):
    """Evaluate model accuracy by comparing predictions to actual labels"""
    print("\n🎯 Evaluating Model Accuracy...")
    
    try:
        # Load actual test data
        with open('real_asteroid_data_test.json', 'r') as f:
            actual_data = json.load(f)
        
        if not test_predictions or not test_predictions.get('predictions'):
            print("   ❌ No test predictions available")
            return False
        
        predictions = test_predictions['predictions']
        
        # Compare predictions vs actual
        correct = 0
        total = min(len(predictions), len(actual_data))
        
        risk_comparison = {}
        
        for i in range(total):
            predicted_risk = predictions[i]['predicted_risk_level']
            actual_risk = actual_data[i]['risk_level']
            
            if predicted_risk == actual_risk:
                correct += 1
            
            # Track per-risk accuracy
            if actual_risk not in risk_comparison:
                risk_comparison[actual_risk] = {'correct': 0, 'total': 0}
            
            risk_comparison[actual_risk]['total'] += 1
            if predicted_risk == actual_risk:
                risk_comparison[actual_risk]['correct'] += 1
        
        overall_accuracy = (correct / total) * 100
        print(f"   Overall Accuracy: {correct}/{total} ({overall_accuracy:.2f}%)")
        
        print(f"   Per-Risk Accuracy:")
        for risk, stats in risk_comparison.items():
            accuracy = (stats['correct'] / stats['total']) * 100
            print(f"     {risk}: {stats['correct']}/{stats['total']} ({accuracy:.1f}%)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Accuracy evaluation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🌟 StarShield API Test Suite")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 0
    
    # Health tests
    total_tests += 1
    if test_health():
        tests_passed += 1
    
    # Model info test
    total_tests += 1
    if test_model_info():
        tests_passed += 1
    
    # Single prediction test
    total_tests += 1
    if test_single_prediction():
        tests_passed += 1
    
    # Batch prediction test
    total_tests += 1
    if test_batch_prediction():
        tests_passed += 1
    
    # Training data test
    total_tests += 1
    if test_train_data_endpoint():
        tests_passed += 1
    
    # Test data test
    total_tests += 1
    test_predictions = test_test_data_endpoint()
    if test_predictions:
        tests_passed += 1
    
    # Accuracy evaluation
    total_tests += 1
    if evaluate_accuracy(test_predictions):
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🏆 Test Results: {tests_passed}/{total_tests} tests passed")
    print("=" * 60)
    
    if tests_passed == total_tests:
        print("✅ All tests passed! The API is working correctly.")
    else:
        print(f"⚠️  {total_tests - tests_passed} tests failed. Check the output above.")

if __name__ == "__main__":
    main()