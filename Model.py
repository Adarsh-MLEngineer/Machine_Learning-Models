import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
Raw_Data = pd.read_csv("Email.csv")

# Display basic information about the dataset
print("Initial Data Info:")
print(Raw_Data.head())
print(Raw_Data.columns)
print(Raw_Data.info())
print(Raw_Data.describe())

# Convert 'Email No.' to numeric values, coercing errors to NaN
Raw_Data['Email No.'] = pd.to_numeric(Raw_Data['Email No.'].str.replace('Email ', ''), errors='coerce')

# Drop rows with NaN values in 'Email No.'
Raw_Data = Raw_Data.dropna(subset=['Email No.'])

# Drop the 'Email No.' column as it's not needed for prediction
Raw_Data = Raw_Data.drop(columns=['Email No.'])

# Features and target variable
X = Raw_Data.drop(columns=['Prediction'])
y = Raw_Data['Prediction']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check the shapes of the resulting datasets
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Initialize and train the model
model = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')

# Plotting the first 20 predictions vs actual values
plt.figure(figsize=(10, 6))
plt.plot(range(20), y_test.head(20).values, marker='o', linestyle='-', color='b', label='Actual')
plt.plot(range(20), predictions[:20], marker='x', linestyle='-', color='r', label='Predicted')
plt.title('Actual vs Predicted Values (First 20 samples)')
plt.xlabel('Sample Index')
plt.ylabel('Prediction')
plt.legend()
plt.grid(True)
plt.show()
