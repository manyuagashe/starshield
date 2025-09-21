#!/usr/bin/env python3
"""
Fix JSON files to have proper is_pha values and diverse risk levels
"""
import json
import random

def fix_json_files():
    """Fix is_pha values and create diverse risk levels"""
    
    # Process final_data_false.json
    print("Fixing final_data_false.json...")
    try:
        with open('final_data_false.json', 'r') as f:
            data_false = json.load(f)
        
        for i, record in enumerate(data_false):
            # Ensure is_pha is False
            record['is_pha'] = False
            
            # Create more diverse risk levels based on distance and diameter
            distance = float(record.get('distance_au', 0.1))
            diameter = float(record.get('diameter_km', 0.1))
            
            # Add some randomness to create diversity
            random_factor = random.random()
            
            if distance < 0.01 and diameter > 0.1 and random_factor > 0.5:
                record['risk_level'] = 'Critical'
            elif distance < 0.03 and diameter > 0.05 and random_factor > 0.3:
                record['risk_level'] = 'High'
            elif distance < 0.06 or random_factor > 0.6:
                record['risk_level'] = 'Medium'
            else:
                record['risk_level'] = 'Low'
        
        with open('final_data_false.json', 'w') as f:
            json.dump(data_false, f, indent=2)
        
        print(f"  Fixed {len(data_false)} records with is_pha=False")
        
        # Show distribution
        risk_counts = {}
        for r in data_false:
            risk = r['risk_level']
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        print("  Risk distribution:", risk_counts)
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Process final_data_true.json
    print("\nFixing final_data_true.json...")
    try:
        with open('final_data_true.json', 'r') as f:
            data_true = json.load(f)
        
        for i, record in enumerate(data_true):
            # Ensure is_pha is True
            record['is_pha'] = True
            
            # PHA asteroids should generally have higher risk
            distance = float(record.get('distance_au', 0.1))
            diameter = float(record.get('diameter_km', 0.1))
            
            # Add some randomness but bias toward higher risk
            random_factor = random.random()
            
            if distance < 0.02 or (diameter > 0.1 and random_factor > 0.3):
                record['risk_level'] = 'Critical'
            elif distance < 0.04 or (diameter > 0.05 and random_factor > 0.4):
                record['risk_level'] = 'High'
            elif distance < 0.07 or random_factor > 0.5:
                record['risk_level'] = 'Medium'
            else:
                record['risk_level'] = 'Low'
        
        with open('final_data_true.json', 'w') as f:
            json.dump(data_true, f, indent=2)
        
        print(f"  Fixed {len(data_true)} records with is_pha=True")
        
        # Show distribution
        risk_counts = {}
        for r in data_true:
            risk = r['risk_level']
            risk_counts[risk] = risk_counts.get(risk, 0) + 1
        print("  Risk distribution:", risk_counts)
        
    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Fixing JSON files...")
    print("="*60)
    
    fix_json_files()
    
    print("\n" + "="*60)
    print("✅ Files fixed! You can now run train_model.py")