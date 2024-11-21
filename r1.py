import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# Load the dataset
file_path = 'accidents_india.csv'  # Replace with your file's location
accident_data = pd.read_csv(file_path)

# Handle missing values
accident_data_cleaned = accident_data.dropna()

# Encode categorical features
label_encoders = {}
categorical_columns = ['Day_of_Week', 'Light_Conditions', 'Accident_Severity']
for col in categorical_columns:
    label_encoders[col] = LabelEncoder()
    accident_data_cleaned[col] = label_encoders[col].fit_transform(accident_data_cleaned[col])

# Split features and target
X = accident_data_cleaned.drop(columns=['Accident_Severity'])
y = accident_data_cleaned['Accident_Severity']

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(rf_model, 'accident_model.pkl')

# Save the label encoders and scaler
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and preprocessing objects saved successfully!")
