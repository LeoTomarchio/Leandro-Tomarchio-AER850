import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score
from sklearn.base import BaseEstimator, ClassifierMixin
from joblib import dump
from scipy.stats import randint
import joblib

class StackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
    
    def fit(self, X, y):
        # Train base models
        for model in self.base_models:
            model.fit(X, y)
        
        # Create meta-features
        meta_features = self._get_meta_features(X)
        
        # Train meta-model
        self.meta_model.fit(meta_features, y)
        
        return self
    
    def predict(self, X):
        meta_features = self._get_meta_features(X)
        return self.meta_model.predict(meta_features)
    
    def _get_meta_features(self, X):
        return np.column_stack([
            model.predict_proba(X) for model in self.base_models
        ])

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

"""Cross Validation for Multiple Models"""
def train_and_evaluate(model_name, X, y, n_folds=5):
    # Create the model based on the model_name
    if model_name == 'logistic':
        model = LogisticRegression(random_state=42, max_iter=1000)
    elif model_name == 'decision_tree':
        model = DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            criterion='gini',
            random_state=42
        )
    elif model_name == 'svm':
        model = SVC(
            C=1.0,
            kernel='rbf',
            gamma='scale',
            class_weight='balanced',
            random_state=42
        )
    else:
        raise ValueError("Invalid model name. Choose 'logistic', 'decision_tree', or 'svm'.")

    # Perform cross-validation
    accuracy_scores = cross_val_score(model, X, y, cv=n_folds, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=n_folds, scoring='f1_weighted')
    precision_scores = cross_val_score(model, X, y, cv=n_folds, scoring='precision_weighted', error_score='raise')

    # Train the model on the full dataset
    model.fit(X, y)

    # Print results
    print(f"\nResults for {model_name.replace('_', ' ').title()} model:")
    print(f"Accuracy scores: {accuracy_scores}")
    print(f"Mean accuracy: {accuracy_scores.mean():.4f} (+/- {accuracy_scores.std() * 2:.4f})")
    print(f"F1 scores: {f1_scores}")
    print(f"Mean F1: {f1_scores.mean():.4f} (+/- {f1_scores.std() * 2:.4f})")
    print(f"Precision scores: {precision_scores}")
    print(f"Mean precision: {precision_scores.mean():.4f} (+/- {precision_scores.std() * 2:.4f})")

    return model  # Return the trained model

# Train and evaluate each model, then save it
logistic_model = train_and_evaluate('logistic', X, y)
joblib.dump(logistic_model, 'logistic_model.joblib')

decision_tree_model = train_and_evaluate('decision_tree', X, y)
joblib.dump(decision_tree_model, 'decision_tree_model.joblib')

svm_model = train_and_evaluate('svm', X, y)
joblib.dump(svm_model, 'svm_model.joblib')

print("All models have been saved successfully.")

"""Random Search CV"""
# Define the parameter distribution
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Set up RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf_classifier,
    param_distributions=param_grid,
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

# Save the model
joblib.dump(random_search, 'random_search.joblib')

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
# Create base models
rf_model = best_rf_model_random  # Use the best Random Forest model from RandomizedSearchCV
lr_model = LogisticRegression(random_state=42)

# Create meta-model
meta_model = LogisticRegression(random_state=42)

# Create and train stacking classifier
stacking_clf = StackingClassifier(base_models=[rf_model, lr_model], meta_model=meta_model)
stacking_clf.fit(X_train, y_train)

# Save the model
joblib.dump(stacking_clf, 'stacking_clf.joblib')

# Make predictions
y_pred_stacking = stacking_clf.predict(X_test)

# Evaluate the stacked model
print("\nStacking Classifier Results:")
print("Accuracy score: {:.2f}".format(accuracy_score(y_test, y_pred_stacking)))
print("\nClassification Report:")
print(classification_report(y_test, y_pred_stacking))

# Generate and plot confusion matrix for stacked model
cm_stacking = confusion_matrix(y_test, y_pred_stacking)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_stacking, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Stacking Classifier')
plt.show()

print("Confusion Matrix for Stacking Classifier:")
print(cm_stacking)
