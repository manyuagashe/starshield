#!/usr/bin/env python3
"""
Script to convert NASA API JSON format to array format with is_pha field
"""
import json

def convert_nasa_json_to_array(input_filename, output_filename, is_pha_value):
    """
    Convert NASA API JSON format to array of objects with is_pha field
    
    Args:
        input_filename: Input JSON file in NASA API format
        output_filename: Output JSON file name
        is_pha_value: Boolean value for is_pha field
    """
    try:
        # Read the NASA format JSON file
        with open(input_filename, 'r') as f:
            nasa_data = json.load(f)
        
        print(f"\nProcessing {input_filename}...")
        
        # Extract fields and data
        fields = nasa_data.get('fields', [])
        raw_data = nasa_data.get('data', [])
        
        print(f"  - Found {len(raw_data)} records")
        print(f"  - Fields: {fields}")
        
        # Create field mapping to expected names
        field_mapping = {
            'des': 'object_name',        # designation
            'cd': 'approach_date',       # close approach date
            'dist': 'distance_au',       # distance in AU
            'v_rel': 'velocity_kms',     # relative velocity
            'diameter': 'diameter_km',   # diameter
            'v_inf': 'v_infinity_kms',   # velocity at infinity
            'orbit_id': 'orbit_class',   # orbit classification
            'h': 'absolute_magnitude'    # absolute magnitude
        }
        
        # Convert to array of objects
        converted_data = []
        
        for record in raw_data:
            # Create object from field names and values
            record_obj = {}
            
            for i, field_name in enumerate(fields):
                if i < len(record):
                    value = record[i]
                    
                    # Map to expected field name
                    if field_name in field_mapping:
                        mapped_name = field_mapping[field_name]
                    else:
                        mapped_name = field_name
                    
                    # Handle data type conversions
                    if mapped_name in ['distance_au', 'velocity_kms', 'diameter_km', 'v_infinity_kms']:
                        # Convert to float, handle null values
                        record_obj[mapped_name] = float(value) if value is not None else 0.1
                    elif mapped_name == 'orbit_class':
                        # Convert orbit_id to standard classes
                        # You may need to adjust this mapping based on your data
                        try:
                            orbit_num = int(value) if value else 0
                            # Simple mapping - adjust as needed
                            if orbit_num < 50:
                                record_obj[mapped_name] = 'APO'
                            elif orbit_num < 100:
                                record_obj[mapped_name] = 'ATE'
                            elif orbit_num < 150:
                                record_obj[mapped_name] = 'AMO'
                            else:
                                record_obj[mapped_name] = 'IEO'
                        except:
                            record_obj[mapped_name] = 'APO'  # default
                    else:
                        record_obj[mapped_name] = value
            
            # Add is_pha field
            record_obj['is_pha'] = is_pha_value
            
            # Add risk level based on distance and size
            distance = record_obj.get('distance_au', 1.0)
            diameter = record_obj.get('diameter_km', 0.1)
            
            if distance < 0.01 and diameter > 0.1:
                record_obj['risk_level'] = 'Critical'
            elif distance < 0.05 and diameter > 0.05:
                record_obj['risk_level'] = 'High'
            elif distance < 0.1:
                record_obj['risk_level'] = 'Medium'
            else:
                record_obj['risk_level'] = 'Low'
            
            # Ensure all required fields exist
            required_fields = ['distance_au', 'velocity_kms', 'diameter_km', 'v_infinity_kms', 'orbit_class']
            for field in required_fields:
                if field not in record_obj:
                    # Set default values
                    if field == 'orbit_class':
                        record_obj[field] = 'APO'
                    else:
                        record_obj[field] = 0.1
            
            converted_data.append(record_obj)
        
        # Save as array
        with open(output_filename, 'w') as f:
            json.dump(converted_data, f, indent=2)
        
        print(f"  - Converted {len(converted_data)} records")
        print(f"  - Added is_pha={is_pha_value} to all records")
        print(f"  - Saved to {output_filename}")
        
        # Show sample record
        if converted_data:
            print(f"\n  Sample converted record:")
            print(f"  {json.dumps(converted_data[0], indent=4)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing '{input_filename}': {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to convert both files"""
    print("🚀 Converting NASA API JSON to array format with is_pha field")
    print("=" * 60)
    
    # Convert both files
    conversions = [
        ('final_data_false.json', 'final_data_false_converted.json', False),
        ('final_data_true.json', 'final_data_true_converted.json', True)
    ]
    
    success_count = 0
    
    for input_file, output_file, is_pha_value in conversions:
        if convert_nasa_json_to_array(input_file, output_file, is_pha_value):
            success_count += 1
    
    print("\n" + "=" * 60)
    
    if success_count == len(conversions):
        print("✅ Successfully converted all files!")
        print("\nConverted files created:")
        print("  - final_data_false_converted.json (with is_pha=false)")
        print("  - final_data_true_converted.json (with is_pha=true)")
        print("\nTo use these files with your training script:")
        print("  1. Rename the converted files:")
        print("     mv final_data_false_converted.json final_data_false.json")
        print("     mv final_data_true_converted.json final_data_true.json")
        print("  2. Run your training script:")
        print("     python train_model.py")
    else:
        print(f"⚠️  Converted {success_count}/{len(conversions)} files")
        print("Please check the errors above.")

if __name__ == "__main__":
    main()