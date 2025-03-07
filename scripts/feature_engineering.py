import pandas as pd
import numpy as np
import argparse
from pathlib import Path


# Load processed datasets
# df_inhaler = pd.read_parquet("../data/processed/inhaler_air_merged/", engine="pyarrow")
# df_patients = pd.read_parquet("../data/raw/iot_inhaler/patients.parquet", engine="pyarrow")

def load_data(inhaler_path, patients_path):
    # Load parquet files
    df_inhaler = pd.read_parquet(f"{inhaler_path}", engine="pyarrow")
    df_patients = pd.read_parquet(f"{patients_path}", engine="pyarrow")
    return df_inhaler, df_patients
    

# Merge and add features to dataset
def create_clinical_features(df_inhaler, df_patients):
    # Merge
    df = pd.merge(df_inhaler, df_patients, on='patient_id', how='left')
    
    # 1. Clinical Temporal Variables
    df['hour'] = df['timestamp'].dt.hour
    df['is_night'] = df['hour'].between(0,5).astype(int)
    df['is_peak_pollution'] = df['hour'].between(7,10).astype(int)
    
    # 2. Environmental Exposures
    df['pm25_24h_avg'] = df.groupby('district')['PM2.5 (µg/m³)'].transform(
        lambda x: x.rolling(24).mean())
    df['no2_exceedance'] = (df['NO2 (µg/m³)'] > 40).astype(int)
    
    # 3. Adherence Patterns
    df['adherence_trend'] = df.groupby('patient_id')['symbicort_adherence'].transform(
        lambda x: x.rolling(72, min_periods=24).mean())
    
    # 4. GEMA-5.0 Risk Score
    severity_weights = {
        'Intermittent': 1.0,
        'Mild Persistent': 1.5,
        'Moderate Persistent': 2.0,
        'Severe Persistent': 3.0
    }
    df['gema_risk_score'] = df['gema_severity'].map(severity_weights) * \
                           (1.2 - df['symbicort_adherence'])
    
    # 5. Geographic Factor
    district_risk = df.groupby('district')['puffs'].mean().to_dict()
    df['district_risk'] = df['district'].map(district_risk)
    
    return df

# Save data
# clinical_df.to_csv('../data/processed/train_ready/clinical_df_v1.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data processed")
    parser.add_argument('--input_inhaler', required=True, help='inhaler raw data')
    parser.add_argument('--input_patients', required=True, help='patients processed data')
    parser.add_argument('--output', required=True, help='output .csv file')
    
    args = parser.parse_args()
    
    try:
        df_inhaler, df_patients = load_data(args.input_inhaler, args.input_patients)
        clinical_df = create_clinical_features(df_inhaler, df_patients)
        clinical_df.to_csv(args.output, index=False)
        print(f"Data saved in {args.output}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise