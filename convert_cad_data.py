#!/usr/bin/env python3
"""
Convert NASA JPL CAD (Close Approach Data) to training format for asteroid risk model.

This script takes the raw CAD data and converts it to the format expected by our 
asteroid risk prediction model, making reasonable assumptions for missing fields.
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any

def estimate_diameter_from_magnitude(h_magnitude: float) -> float:
    """
    Estimate diameter from absolute magnitude using the standard formula.
    
    Formula: D = 1329 / sqrt(albedo) * 10^(-0.2 * H)
    Assuming typical albedo of 0.14 for most asteroids
    """
    if pd.isna(h_magnitude):
        return 0.1  # Default small size
    
    albedo = 0.14  # Typical asteroid albedo
    diameter_km = 1329 / np.sqrt(albedo) * (10 ** (-0.2 * float(h_magnitude)))
    return round(diameter_km, 3)

def classify_as_pha(distance_au: float, diameter_km: float) -> bool:
    """
    Classify as Potentially Hazardous Asteroid (PHA) based on NASA criteria:
    - Minimum orbit intersection distance (MOID) with Earth < 0.05 AU
    - Diameter > 140 meters (0.14 km)
    
    Since we don't have MOID, we'll use approach distance as approximation.
    """
    return distance_au < 0.05 and diameter_km > 0.14

def estimate_orbit_class(distance_au: float, velocity_kms: float) -> str:
    """
    Estimate orbit class based on approach characteristics.
    This is a simplified heuristic - real classification requires orbital elements.
    
    - APO (Apollo): Earth-crossing, typically higher velocities
    - ATE (Aten): Earth-crossing, smaller orbits
    - AMO (Amor): Earth-approaching, don't cross Earth's orbit
    - IEO (Interior Earth Object): Inside Earth's orbit
    """
    if distance_au < 0.015 and velocity_kms > 20:
        return "APO"  # High speed, very close approach
    elif distance_au < 0.02 and velocity_kms > 15:
        return "ATE"  # Close approach, moderate speed
    elif distance_au < 0.05:
        return "AMO"  # Moderately close approach
    else:
        return "AMO"  # Default to AMO for distant approaches

def calculate_risk_level(distance_au: float, velocity_kms: float, diameter_km: float, is_pha: bool) -> str:
    """
    Calculate risk level based on multiple factors.
    This is a heuristic approach - real risk assessment is much more complex.
    """
    # Base risk score (0-100)
    risk_score = 0
    
    # Distance factor (closer = higher risk)
    if distance_au < 0.01:
        risk_score += 40
    elif distance_au < 0.02:
        risk_score += 30
    elif distance_au < 0.05:
        risk_score += 20
    else:
        risk_score += 10
    
    # Velocity factor (faster = higher risk)
    if velocity_kms > 25:
        risk_score += 30
    elif velocity_kms > 20:
        risk_score += 25
    elif velocity_kms > 15:
        risk_score += 20
    else:
        risk_score += 10
    
    # Size factor (larger = higher risk)
    if diameter_km > 1.0:
        risk_score += 20
    elif diameter_km > 0.5:
        risk_score += 15
    elif diameter_km > 0.2:
        risk_score += 10
    else:
        risk_score += 5
    
    # PHA bonus
    if is_pha:
        risk_score += 10
    
    # Convert to categorical risk
    if risk_score >= 80:
        return "Critical"
    elif risk_score >= 60:
        return "High"
    elif risk_score >= 40:
        return "Medium"
    else:
        return "Low"

def convert_cad_to_training_data(cad_file_path: str, output_file_path: str, sample_size: int = 1300, train_split: float = 0.8) -> Dict[str, Any]:
    """
    Convert CAD JSON data to training format with sampling and train/test split.
    
    Args:
        cad_file_path: Path to the CAD JSON file
        output_file_path: Base path for output files (will create multiple files)
        sample_size: Number of records to sample from the full dataset
        train_split: Fraction of sampled data to use for training (rest for testing)
    """
    # Load CAD data
    with open(cad_file_path, 'r') as f:
        cad_data = json.load(f)
    
    # Create DataFrame
    full_df = pd.DataFrame(cad_data['data'], columns=cad_data['fields'])
    
    # Sample random subset
    if len(full_df) > sample_size:
        sampled_df = full_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        print(f"Sampled {sample_size} records from {len(full_df)} total records")
    else:
        sampled_df = full_df
        print(f"Using all {len(full_df)} records (less than requested sample size)")
    
    df = sampled_df
    
    # Convert string numbers to float
    numeric_columns = ['dist', 'dist_min', 'dist_max', 'v_rel', 'v_inf', 'h']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create training data
    training_records = []
    
    for _, row in df.iterrows():
        # Extract base features
        distance_au = float(row['dist'])
        velocity_kms = float(row['v_rel'])
        v_infinity_kms = float(row['v_inf'])
        
        # Estimate diameter from magnitude
        diameter_km = estimate_diameter_from_magnitude(row['h'])
        
        # Classify as PHA
        is_pha = classify_as_pha(distance_au, diameter_km)
        
        # Estimate orbit class
        orbit_class = estimate_orbit_class(distance_au, velocity_kms)
        
        # Calculate risk level
        risk_level = calculate_risk_level(distance_au, velocity_kms, diameter_km, is_pha)
        
        # Create training record
        record = {
            "object_name": row['des'],
            "approach_date": row['cd'],
            "distance_au": round(distance_au, 6),
            "velocity_kms": round(velocity_kms, 2),
            "diameter_km": diameter_km,
            "v_infinity_kms": round(v_infinity_kms, 2),
            "is_pha": bool(is_pha),  # Ensure it's a Python bool
            "orbit_class": orbit_class,
            "risk_level": risk_level,
            "absolute_magnitude": float(row['h']) if pd.notna(row['h']) else None,
            "source": "NASA JPL CAD API"
        }
        
        training_records.append(record)
    
    # Split into train and test sets
    train_size = int(len(training_records) * train_split)
    
    # Shuffle data for random split
    import random
    random.seed(42)
    random.shuffle(training_records)
    
    train_data = training_records[:train_size]
    test_data = training_records[train_size:]
    
    # Save files
    base_path = output_file_path.rsplit('.', 1)[0]  # Remove .json extension
    
    train_file = f"{base_path}_train.json"
    test_file = f"{base_path}_test.json"
    full_file = output_file_path  # Keep original filename for backward compatibility
    
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    with open(full_file, 'w') as f:
        json.dump(training_records, f, indent=2)
    
    # Generate statistics
    stats = {
        "total_records": len(training_records),
        "train_records": len(train_data),
        "test_records": len(test_data),
        "train_file": train_file,
        "test_file": test_file,
        "full_file": full_file,
        "risk_distribution": {},
        "orbit_class_distribution": {},
        "pha_count": sum(1 for r in training_records if r['is_pha'])
    }
    
    # Risk level distribution
    for record in training_records:
        risk = record['risk_level']
        stats['risk_distribution'][risk] = stats['risk_distribution'].get(risk, 0) + 1
    
    # Orbit class distribution
    for record in training_records:
        orbit = record['orbit_class']
        stats['orbit_class_distribution'][orbit] = stats['orbit_class_distribution'].get(orbit, 0) + 1
    
    return stats

if __name__ == "__main__":
    print("🚀 Converting NASA JPL CAD data to training format...")
    
    # Convert the data with sampling and train/test split
    stats = convert_cad_to_training_data(
        'raw_data/cad.json',
        'real_asteroid_data.json',
        sample_size=1300,
        train_split=0.8
    )
    
    print(f"✅ Conversion complete!")
    print(f"📊 Generated {stats['total_records']} training records")
    print(f"📈 Split into:")
    print(f"   Training: {stats['train_records']} records ({stats['train_records']/stats['total_records']*100:.1f}%)")
    print(f"   Testing: {stats['test_records']} records ({stats['test_records']/stats['total_records']*100:.1f}%)")
    
    print(f"\n📈 Risk Level Distribution:")
    for risk, count in stats['risk_distribution'].items():
        percentage = (count / stats['total_records']) * 100
        print(f"   {risk}: {count} ({percentage:.1f}%)")
    
    print(f"🛸 Orbit Class Distribution:")
    for orbit, count in stats['orbit_class_distribution'].items():
        percentage = (count / stats['total_records']) * 100
        print(f"   {orbit}: {count} ({percentage:.1f}%)")
    
    print(f"⚠️  Potentially Hazardous Asteroids: {stats['pha_count']}")
    
    print(f"\n🎯 Files created:")
    print(f"   Training data: {stats['train_file']}")
    print(f"   Test data: {stats['test_file']}")
    print(f"   Full dataset: {stats['full_file']}")
    print(f"\n🔥 Ready to train your model with: python train_model.py")