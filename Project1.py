import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score
from scipy.stats import randint

"""Data Processing"""
df = pd.read_csv('Project_1_Data.csv')

"""Data Visualization"""
# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Extract X, Y, Z coordinates from the dataframe
X = df['X']
Y = df['Y']
Z = df['Z']

# Create the scatter plot
scatter = ax.scatter(X, Y, Z, c=df['Step'], cmap='viridis')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot of Data')

# Add a color bar
plt.colorbar(scatter, label='Step')

# Show the plot
plt.show()

"""Correlation Matrix"""
# Calculate the correlation matrix
correlation_matrix = df.corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(abs(correlation_matrix), annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Pearson Correlation Matrix')
plt.tight_layout()
plt.show()

"""Data Splitting"""
X = df.drop('Step', axis=1)
y = df['Step']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# """Logistic Cross Validation"""
# model = LogisticRegression()

# # Number of folds
# n_folds = 5

# # Perform cross-validation
# accuracy_scores = cross_val_score(model, X, y, cv=n_folds, scoring='accuracy')
# f1_scores = cross_val_score(model, X, y, cv=n_folds, scoring='f1_weighted')
# precision_scores = cross_val_score(model, X, y, cv=n_folds, scoring='precision_weighted')

# # Print results
# print(f"Accuracy scores: {accuracy_scores}")
# print(f"Mean accuracy: {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std() * 2:.4f})")
# print(f"F1 scores: {f1_scores}")
# print(f"Mean F1: {f1_scores.mean():.4f} (+/- {f1_scores.std() * 2:.4f})")
# print(f"Precision scores: {precision_scores}")
# print(f"Mean precision: {precision_scores.mean():.4f} (+/- {precision_scores.std() * 2:.4f})")

"""Random Search CV"""
# Define the parameter distribution
param_dist = {
    'n_estimators': randint(100, 500),
    'max_depth': [None] + list(randint(10, 50).rvs(4)),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'max_features': ['auto', 'sqrt', 'log2']
}

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf_classifier,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

# Fit the random search to the data
random_search.fit(X_train, y_train)

# Print the best parameters and score
print("Best parameters found: ", random_search.best_params_)
print("Best cross-validation score: {:.2f}".format(random_search.best_score_))

# Get the best model
best_rf_model_random = random_search.best_estimator_

# Make predictions on the test set
y_pred_random = best_rf_model_random.predict(X_test)

# Print the accuracy score and classification report
print("Accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred_random)))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_random))

"""Confusion Matrix"""
# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred_random)

# Create a figure and axis
plt.figure(figsize=(10, 8))

# Plot the confusion matrix using seaborn
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Set labels and title
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Random Forest Classifier')

# Show the plot
plt.show()

# Print the confusion matrix
print("Confusion Matrix:")
print(cm)


"""Model Stacking"""

"""Joblib"""