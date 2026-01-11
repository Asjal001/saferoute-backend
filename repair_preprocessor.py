# repair_preprocessor.py
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

print("Repairing preprocessor for local compatibility...")

# 1. Load the dataset (We need it to learn the scaling math again)
try:
    df = pd.read_csv('Synthetic_Transportation_Dataset_Expanded_v2.csv')
    
    # 2. Re-create the derived time features
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], dayfirst=True)
    df['Hour'] = df['Timestamp'].dt.hour
    df['DayOfWeek'] = df['Timestamp'].dt.dayofweek

    # 3. Define the exact same Preprocessor as before
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), ['Hour', 'DayOfWeek', 'Latitude', 'Longitude', 'Avg_Speed(km/h)']),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['Road_ID', 'Weather'])
        ]
    )

    # 4. Fit it to the data
    # We select the columns needed for training
    X = df[['Hour', 'DayOfWeek', 'Latitude', 'Longitude', 'Avg_Speed(km/h)', 'Road_ID', 'Weather']]
    preprocessor.fit(X)

    # 5. Save it freshly using YOUR local Scikit-Learn version
    joblib.dump(preprocessor, 'preprocessor.pkl')
    
    print("✅ SUCCESS: 'preprocessor.pkl' has been rebuilt for your computer!")
    print("You can now run 'python app.py' without errors.")

except FileNotFoundError:
    print("❌ ERROR: Could not find the CSV file.")
    print("Please make sure 'Synthetic_Transportation_Dataset_Expanded_v2.csv' is inside this folder.")