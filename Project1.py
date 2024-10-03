import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

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
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Pearson Correlation Matrix')
plt.tight_layout()
plt.show()

"""Data Splitting"""
X = df.drop('Step', axis=1)
y = df['Step']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Grid Search"""

"""Random Search CV"""

"""Confusion Matrix"""

"""Model Stacking"""

"""Joblib"""