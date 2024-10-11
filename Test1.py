import joblib
import numpy as np

# Load the saved logistic regression model
loaded_model = joblib.load('random_search.joblib')

# Create a numpy array with the new data points
new_data = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0, 3.0625, 1.93],
    [9.4, 3, 1.8],
    [9.4, 3, 1.3]
])

# Make predictions
predictions = loaded_model.predict(new_data)

# Print the predictions
for i, pred in enumerate(predictions):
    print(f"Data point {i+1}: {new_data[i]} - Predicted class: {pred}")

# If you want to get probability estimates (if the model supports it)
if hasattr(loaded_model, 'predict_proba'):
    probabilities = loaded_model.predict_proba(new_data)
    for i, prob in enumerate(probabilities):
        print(f"Data point {i+1}: Probability estimates: {prob}")