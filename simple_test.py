#!/usr/bin/env python3
"""
Enhanced debug script for StarShield API data issues
"""
import requests
import json
from api_query import fetch_asteroid_data, process_api_data, get_train_test_data

def debug_api_data_structure():
    """Debug the actual data structure from the API"""
    print("🔍 Debugging API Data Structure...\n")
    
    # Step 1: Check raw API response
    print("1. Fetching raw API response...")
    try:
        # Define the URL
        base_url = "https://ssd-api.jpl.nasa.gov/cad.api?diameter=true&date-min=2025-09-21&date-max=2025-12-21"
        
        # Test with non-PHA asteroids
        non_pha_url = f"{base_url}&pha=false"
        raw_response = fetch_asteroid_data(non_pha_url)  # Pass the URL!
        
        print("✅ API call successful!")
        
        # Show API metadata
        print(f"\nAPI Response Structure:")
        print(f"- Signature: {raw_response.get('signature', {})}")
        print(f"- Count: {raw_response.get('count', 'N/A')}")
        print(f"- Fields: {raw_response.get('fields', [])}")
        
        # Show field indices
        fields = raw_response.get('fields', [])
        print(f"\nField Mapping (index: name):")
        for i, field in enumerate(fields):
            print(f"  {i}: {field}")
            
    except Exception as e:
        print(f"❌ Failed to fetch API data: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Step 2: Check processed data
    print("\n2. Processing API data...")
    try:
        processed_data = process_api_data(raw_response, is_pha=False)  # Note: process_api_data might need is_pha parameter
        print(f"✅ Processed {len(processed_data)} records")
        
        if processed_data:
            print(f"\nFirst processed record:")
            print(json.dumps(processed_data[0], indent=2))
            
            print(f"\nAll fields in processed record:")
            print(list(processed_data[0].keys()))
            
    except Exception as e:
        print(f"❌ Failed to process data: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 3: Test train/test split
    print("\n3. Testing train/test data split...")
    try:
        # This function doesn't need parameters as it has defaults
        train_data, test_data = get_train_test_data()
        print(f"✅ Train/test split successful!")
        print(f"- Training samples: {len(train_data)}")
        print(f"- Test samples: {len(test_data)}")
        
        if train_data:
            print(f"\nFirst training record:")
            print(json.dumps(train_data[0], indent=2))
            
    except Exception as e:
        print(f"❌ Failed to split data: {e}")
        import traceback
        traceback.print_exc()

def debug_backend_endpoint():
    """Debug the backend /data/train endpoint"""
    print("\n\n4. Testing Backend Endpoint...")
    
    # First, let's make a simple health check
    try:
        health_response = requests.get(f"http://localhost:8000/")
        print(f"✅ Server is running: {health_response.json()}")
    except:
        print("❌ Server is not running!")
        return
    
    # Now test the problematic endpoint
    print("\nTesting /data/train endpoint...")
    try:
        response = requests.get("http://localhost:8000/data/train")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error Response: {response.text}")
            
            # Get server logs hint
            print("\n💡 Check your server console for detailed error logs!")
            print("The server should show the exact line where the KeyError occurs.")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

def create_fallback_data():
    """Create fallback JSON files with correct structure"""
    print("\n\n5. Creating Fallback JSON Files...")
    
    try:
        # Try to get data from API with correct structure
        train_data, test_data = get_train_test_data()
        
        # Save to JSON files
        with open('real_asteroid_data_train.json', 'w') as f:
            json.dump(train_data, f, indent=2)
            
        with open('real_asteroid_data_test.json', 'w') as f:
            json.dump(test_data, f, indent=2)
            
        print("✅ Created JSON files from API data!")
        print(f"- real_asteroid_data_train.json: {len(train_data)} records")
        print(f"- real_asteroid_data_test.json: {len(test_data)} records")
        
    except Exception as e:
        print(f"❌ Failed to create JSON files: {e}")
        
        # Create minimal working files
        print("\nCreating minimal working JSON files...")
        
        sample_data = [{
            "object_name": f"TEST-{i}",
            "distance_au": 0.01 + (i * 0.001),
            "velocity_kms": 10.0 + (i * 0.5),
            "diameter_km": 0.02 + (i * 0.002),
            "v_infinity_kms": 12.0 + (i * 0.3),
            "is_pha": i % 3 == 0,
            "orbit_class": ["AMO", "APO", "ATE", "IEO"][i % 4],
            "approach_date": f"2025-09-{15 + (i % 5)}T12:00:00Z",
            "risk_level": ["Low", "Medium", "High", "Critical"][i % 4]
        } for i in range(100)]
        
        train_data = sample_data[:80]
        test_data = sample_data[80:]
        
        with open('real_asteroid_data_train.json', 'w') as f:
            json.dump(train_data, f, indent=2)
            
        with open('real_asteroid_data_test.json', 'w') as f:
            json.dump(test_data, f, indent=2)
            
        print("✅ Created minimal working JSON files!")

if __name__ == "__main__":
    print("🚀 StarShield API Data Debugging Tool\n")
    print("=" * 60)
    
    # Run all debug steps
    debug_api_data_structure()
    debug_backend_endpoint()
    create_fallback_data()
    
    print("\n" + "=" * 60)
    print("\n📋 Next Steps:")
    print("1. Check the output above to see what fields are missing")
    print("2. Look at your server console for detailed error messages")
    print("3. If JSON files were created, restart your server and test again")
    print("4. Run: python simple_test.py")