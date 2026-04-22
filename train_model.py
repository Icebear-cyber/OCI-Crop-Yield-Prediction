# OCI Crop Yield Prediction - Training Script
# Machine Learning model for predicting crop yields

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
from datetime import datetime

# Sample feature columns for crop yield prediction
FEATURE_COLUMNS = [
    'temperature_avg',      # Average temperature (°C)
    'rainfall_mm',          # Total rainfall (mm)
    'humidity_avg',         # Average humidity (%)
    'soil_ph',              # Soil pH level
    'nitrogen_ppm',         # Nitrogen content (ppm)
    'phosphorus_ppm',       # Phosphorus content (ppm)
    'potassium_ppm',        # Potassium content (ppm)
    'area_hectares',        # Farm area (hectares)
    'elevation_m',          # Elevation (meters)
    'irrigation_days',      # Days of irrigation
]

TARGET = 'yield_tons_per_hectare'

def generate_sample_data(n_samples=1000):
    """
    Generate synthetic crop yield data for demonstration
    In production, replace with real agricultural data
    """
    np.random.seed(42)
    
    data = {
        'temperature_avg': np.random.uniform(15, 35, n_samples),
        'rainfall_mm': np.random.uniform(400, 1200, n_samples),
        'humidity_avg': np.random.uniform(50, 90, n_samples),
        'soil_ph': np.random.uniform(5.5, 7.5, n_samples),
        'nitrogen_ppm': np.random.uniform(20, 80, n_samples),
        'phosphorus_ppm': np.random.uniform(15, 60, n_samples),
        'potassium_ppm': np.random.uniform(100, 300, n_samples),
        'area_hectares': np.random.uniform(1, 50, n_samples),
        'elevation_m': np.random.uniform(100, 1500, n_samples),
        'irrigation_days': np.random.randint(30, 150, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate target variable with realistic relationships
    df[TARGET] = (
        2.5 +
        0.05 * df['temperature_avg'] +
        0.003 * df['rainfall_mm'] +
        0.02 * df['humidity_avg'] +
        0.4 * df['soil_ph'] +
        0.015 * df['nitrogen_ppm'] +
        0.01 * df['phosphorus_ppm'] +
        0.005 * df['potassium_ppm'] +
        0.02 * df['area_hectares'] -
        0.0005 * df['elevation_m'] +
        0.01 * df['irrigation_days'] +
        np.random.normal(0, 0.5, n_samples)  # Add noise
    )
    
    return df

def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and return the best one"""
    
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R2:   {r2:.4f}")
    
    # Select best model based on R2 score
    best_model_name = max(results, key=lambda x: results[x]['r2'])
    best_model = results[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    return best_model['model'], results

def save_model_and_metadata(model, scaler, results, feature_names):
    """Save trained model, scaler, and metadata"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_filename = f'crop_yield_model_{timestamp}.pkl'
    joblib.dump(model, model_filename)
    print(f"\nModel saved: {model_filename}")
    
    # Save scaler
    scaler_filename = f'scaler_{timestamp}.pkl'
    joblib.dump(scaler, scaler_filename)
    print(f"Scaler saved: {scaler_filename}")
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'feature_names': feature_names,
        'model_type': type(model).__name__,
        'metrics': {k: {metric: float(v) for metric, v in res.items() if metric != 'model'} 
                   for k, res in results.items()},
    }
    
    metadata_filename = f'model_metadata_{timestamp}.json'
    with open(metadata_filename, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_filename}")
    
    return model_filename, scaler_filename, metadata_filename

def main():
    print("="*60)
    print("  OCI Crop Yield Prediction - Model Training")
    print("="*60)
    
    # Generate or load data
    print("\nGenerating sample data...")
    df = generate_sample_data(n_samples=1000)
    print(f"Dataset shape: {df.shape}")
    print(f"\nSample statistics:\n{df.describe()}")
    
    # Prepare features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")
    
    # Scale features
    print("\nScaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print("\n" + "="*60)
    print("  Training Models")
    print("="*60)
    best_model, results = train_models(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # Save model
    print("\n" + "="*60)
    print("  Saving Model")
    print("="*60)
    model_file, scaler_file, metadata_file = save_model_and_metadata(
        best_model, scaler, results, FEATURE_COLUMNS
    )
    
    print("\n" + "="*60)
    print("  Training Complete!")
    print("="*60)
    print(f"\nUse the saved model for predictions:")
    print(f"  model = joblib.load('{model_file}')")
    print(f"  scaler = joblib.load('{scaler_file}')")

if __name__ == '__main__':
    main()
