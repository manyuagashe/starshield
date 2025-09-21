# api_query.py
import requests
import json
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
import random

def fetch_asteroid_data(url: str) -> Dict[str, Any]:
    """
    Fetch asteroid data from NASA's Small-Body Database API
    
    Args:
        url: API endpoint URL with parameters
        
    Returns:
        Dictionary containing the API response data
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        raise

def process_api_data(api_response: Dict[str, Any], is_pha: bool) -> List[Dict[str, Any]]:
    """
    Process the raw API response into a format suitable for training
    
    Args:
        api_response: Raw API response
        is_pha: Boolean indicating if this data is for PHAs
    """
    if 'data' not in api_response:
        raise ValueError("Invalid API response format: 'data' field not found")
    
    fields = api_response.get('fields', [])
    raw_data = api_response['data']
    
    # Only print fields once
    if is_pha:
        print("\n=== API Fields ===")
        for i, field in enumerate(fields):
            print(f"{i}: {field}")
        print("==================\n")
    
    field_mapping = {
        'des': 'object_name',
        'cd': 'approach_date',
        'dist': 'distance_au',
        'v_rel': 'velocity_kms',
        'diameter': 'diameter_km',
        'v_inf': 'v_infinity_kms',
        'orbit_id': 'orbit_class',
    }
    
    processed_data = []
    for record in raw_data:
        record_dict = {}
        
        for i, value in enumerate(record):
            if i < len(fields):
                field_name = fields[i]
                record_dict[field_name] = value
        
        mapped_record = {}
        for api_field, expected_field in field_mapping.items():
            if api_field in record_dict:
                value = record_dict[api_field]
                
                if expected_field == 'distance_au' and value:
                    mapped_record[expected_field] = float(value)
                elif expected_field == 'velocity_kms' and value:
                    mapped_record[expected_field] = float(value)
                elif expected_field == 'diameter_km' and value:
                    mapped_record[expected_field] = float(value) if value else 0.1
                elif expected_field == 'v_infinity_kms' and value:
                    mapped_record[expected_field] = float(value)
                else:
                    mapped_record[expected_field] = value
        
        # Set is_pha based on the parameter passed
        mapped_record['is_pha'] = is_pha
        
        # Add default values for missing required fields
        if 'diameter_km' not in mapped_record or not mapped_record['diameter_km']:
            mapped_record['diameter_km'] = 0.1
        
        if 'orbit_class' not in mapped_record or not mapped_record['orbit_class']:
            # Try to convert orbit_id to standard classes
            orbit_id = record_dict.get('orbit_id', '')
            try:
                orbit_num = int(orbit_id) if orbit_id else 0
                if orbit_num < 50:
                    mapped_record['orbit_class'] = 'APO'
                elif orbit_num < 100:
                    mapped_record['orbit_class'] = 'ATE'
                elif orbit_num < 150:
                    mapped_record['orbit_class'] = 'AMO'
                else:
                    mapped_record['orbit_class'] = 'IEO'
            except:
                mapped_record['orbit_class'] = 'APO'
        
        valid_orbit_classes = ['ATE', 'APO', 'AMO', 'IEO']
        if mapped_record.get('orbit_class') not in valid_orbit_classes:
            mapped_record['orbit_class'] = 'APO'
        
        # Risk level calculation with some variation based on is_pha
        distance = mapped_record.get('distance_au', 1.0)
        diameter = mapped_record.get('diameter_km', 0.1)
        
        # Add slight bias for PHAs to have higher risk
        bias = 0.02 if is_pha else 0
        
        if distance < (0.01 + bias) and diameter > 0.1:
            risk_level = 'Critical'
        elif distance < (0.05 + bias) and diameter > 0.05:
            risk_level = 'High'
        elif distance < (0.1 + bias):
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        mapped_record['risk_level'] = risk_level
        
        processed_data.append(mapped_record)
    
    return processed_data

def get_train_test_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Fetch asteroid data from the API (both PHA and non-PHA) and split it into training and test sets
    
    Args:
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (train_data, test_data)
    """
    base_url = "https://ssd-api.jpl.nasa.gov/cad.api?diameter=true&date-min=2025-09-21&date-max=2025-12-21"
    
    print("Fetching asteroid data from NASA API...")
    
    # Fetch non-PHA data
    print("\n1. Fetching non-PHA asteroids...")
    non_pha_url = f"{base_url}&pha=false"
    non_pha_response = fetch_asteroid_data(non_pha_url)
    non_pha_data = process_api_data(non_pha_response, is_pha=False)
    print(f"   Fetched {len(non_pha_data)} non-PHA asteroids")
    
    # Fetch PHA data
    print("\n2. Fetching PHA asteroids...")
    pha_url = f"{base_url}&pha=true"
    pha_response = fetch_asteroid_data(pha_url)
    pha_data = process_api_data(pha_response, is_pha=True)
    print(f"   Fetched {len(pha_data)} PHA asteroids")
    
    # Combine both datasets
    all_data = non_pha_data + pha_data
    
    # Shuffle the combined data
    random.seed(random_state)
    random.shuffle(all_data)
    
    print(f"\n3. Combined and shuffled data:")
    print(f"   Total asteroids: {len(all_data)}")
    
    # Show distribution
    pha_count = sum(1 for record in all_data if record['is_pha'])
    non_pha_count = len(all_data) - pha_count
    print(f"   PHA distribution:")
    print(f"     - PHA asteroids: {pha_count} ({pha_count/len(all_data)*100:.1f}%)")
    print(f"     - Non-PHA asteroids: {non_pha_count} ({non_pha_count/len(all_data)*100:.1f}%)")
    
    # Show risk level distribution
    risk_counts = {}
    for record in all_data:
        risk = record['risk_level']
        risk_counts[risk] = risk_counts.get(risk, 0) + 1
    
    print(f"   Risk level distribution:")
    for risk in ['Critical', 'High', 'Medium', 'Low']:
        if risk in risk_counts:
            count = risk_counts[risk]
            print(f"     - {risk}: {count} ({count/len(all_data)*100:.1f}%)")
    
    # Split into train and test sets
    train_data, test_data = train_test_split(
        all_data, 
        test_size=test_size, 
        random_state=random_state,
        stratify=[f"{r['is_pha']}_{r['risk_level']}" for r in all_data]  # Stratify by both is_pha and risk_level
    )
    
    print(f"\n4. Train/Test split:")
    print(f"   Training set: {len(train_data)} samples")
    print(f"   Test set: {len(test_data)} samples")
    
    return train_data, test_data

# Example usage if running this file directly
if __name__ == "__main__":
    try:
        train_data, test_data = get_train_test_data()
        print(f"\nSuccessfully fetched and processed data!")
        
        if train_data:
            print(f"\nSample training record:")
            sample_record = train_data[0]
            for field, value in sample_record.items():
                print(f"  {field}: {value}")
            
            print(f"\nTotal fields per record: {len(sample_record)}")
            
            # Show distribution in training set
            train_pha = sum(1 for r in train_data if r['is_pha'])
            test_pha = sum(1 for r in test_data if r['is_pha'])
            
            print(f"\nPHA distribution in splits:")
            print(f"  Training: {train_pha}/{len(train_data)} PHAs ({train_pha/len(train_data)*100:.1f}%)")
            print(f"  Test: {test_pha}/{len(test_data)} PHAs ({test_pha/len(test_data)*100:.1f}%)")
            
    except Exception as e:
        print(f"Error: {e}")