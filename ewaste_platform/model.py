import pandas as pd
import numpy as np
import os
import joblib
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

os.makedirs('models', exist_ok=True)
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, 'dataset', 'e-waste_final.csv')

# Load the dataset
df = pd.read_csv(file_path)

# Define features (X) and target (y)
features = [
    'state', 'locality_type', 'household_size', 'income_bracket', 'e-literacy_level',
    'total_devices_owned', 'avg_device_age_years', 'broken_devices_stored',
    'upgrade_tendency', 'disposal_method', 'recycling_awareness'
]
target = 'ewaste_kg_per_year'

X = df[features]
y = df[target]

# Define preprocessing for categorical features
categorical_features = [
    'state', 'locality_type', 'income_bracket', 'e-literacy_level',
    'upgrade_tendency', 'disposal_method', 'recycling_awareness'
]
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Create a preprocessor object using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# Create the Random Forest Regressor model pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('scaler', StandardScaler(with_mean=False)), 
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])



# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)


# Train the model
print("Training the Random Forest model...")
model_pipeline.fit(X_train, y_train)
print("Model training complete.")


# Evaluate the model
y_pred = model_pipeline.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"Model Performance on Test Set:")
print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")

# Save the trained model
joblib.dump(model_pipeline, 'models/ewaste_predictor.joblib')
print("Model saved to models/ewaste_predictor.joblib")