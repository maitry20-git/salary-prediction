import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = pd.read_csv('hiring.csv')

# Handle missing values
dataset['experience'].fillna(0, inplace=True)
dataset['test_score(out of 10)'].fillna(dataset['test_score(out of 10)'].mean(), inplace=True)

# Convert words to integer values
def convert_to_int(word):
    word_dict = {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
                 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11,
                 'twelve': 12, 'zero': 0, 0: 0}
    return word_dict.get(word, 0)  # Defaults to 0 for unknown words

dataset['experience'] = dataset['experience'].apply(convert_to_int)

# Define features and target variable
X = dataset.iloc[:, :-1].copy()
y = dataset.iloc[:, -1]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train)

# Evaluate model
print("Model Accuracy:", regressor.score(X_test_scaled, y_test))

# Save both the model and scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump(regressor, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

# Load model and scaler
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# Predict new data
scaled_input = loaded_scaler.transform([[2, 9, 6]])
print(model.predict(scaled_input))
