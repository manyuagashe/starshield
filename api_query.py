# api_query.py
import requests
import json
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split

def fetch_asteroid_data(url: str = "https://ssd-api.jpl.nasa.gov/cad.api?diameter=true&date-min=2025-09-11&date-max=2025-09-21&pha=true") -> Dict[str, Any]:
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

def process_api_data(api_response: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Process the raw API response into a format suitable for training
    """
    if 'data' not in api_response:
        raise ValueError("Invalid API response format: 'data' field not found")
    
    fields = api_response.get('fields', [])
    raw_data = api_response['data']
    
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
        'pha': 'is_pha',
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
                
                # FORCE is_pha to be True for all records
                if expected_field == 'is_pha':
                    mapped_record[expected_field] = True  # Always True
                elif expected_field == 'distance_au' and value:
                    mapped_record[expected_field] = float(value)
                elif expected_field == 'velocity_kms' and value:
                    mapped_record[expected_field] = float(value)
                elif expected_field == 'diameter_km' and value:
                    mapped_record[expected_field] = float(value) if value else 0.1
                elif expected_field == 'v_infinity_kms' and value:
                    mapped_record[expected_field] = float(value)
                else:
                    mapped_record[expected_field] = value
        
        # Ensure is_pha is always present and True
        mapped_record['is_pha'] = True
        
        # Add default values for missing required fields
        if 'diameter_km' not in mapped_record or not mapped_record['diameter_km']:
            mapped_record['diameter_km'] = 0.1
        
        if 'orbit_class' not in mapped_record or not mapped_record['orbit_class']:
            mapped_record['orbit_class'] = 'APO'
        
        valid_orbit_classes = ['ATE', 'APO', 'AMO', 'IEO']
        if mapped_record.get('orbit_class') not in valid_orbit_classes:
            mapped_record['orbit_class'] = 'APO'
        
        # Risk level calculation
        distance = mapped_record.get('distance_au', 1.0)
        diameter = mapped_record.get('diameter_km', 0.1)
        
        if distance < 0.01 and diameter > 0.1:
            risk_level = 'Critical'
        elif distance < 0.05 and diameter > 0.05:
            risk_level = 'High'
        elif distance < 0.1:
            risk_level = 'Medium'
        else:
            risk_level = 'Low'
        
        mapped_record['risk_level'] = risk_level
        
        processed_data.append(mapped_record)
    
    return processed_data


def get_train_test_data(test_size: float = 0.2, random_state: int = 42) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Fetch asteroid data from the API and split it into training and test sets
    
    Args:
        test_size: Proportion of data to use for testing (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (train_data, test_data)
    """
    # Fetch data from API
    api_response = fetch_asteroid_data()
    
    # Process the data
    processed_data = process_api_data(api_response)
    
    # Split into train and test sets
    train_data, test_data = train_test_split(
        processed_data, 
        test_size=test_size, 
        random_state=random_state
    )
    
    return train_data, test_data

# Example usage if running this file directly
if __name__ == "__main__":
    try:
        train_data, test_data = get_train_test_data()
        print(f"\nSuccessfully fetched data:")
        print(f"Training samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")
        
        if train_data:
            print(f"\nSample training record with all fields:")
            for field, value in train_data[0].items():
                print(f"  {field}: {value}")
            
            print(f"\nTotal fields per record: {len(train_data[0])}")
    except Exception as e:
        print(f"Error: {e}")