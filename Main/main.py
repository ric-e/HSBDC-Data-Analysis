import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data
X1 = [17.4,19,16.8,18,25.8,24.4,26.7,25,26.6,18.5,20.7,16.6,22.5,24.5,22.1,19.7,18.3,18.1,17.8,25.1,24.5,26.4,24.3,19.3,17,19,17.4,18.3,16,25.1,24.5,19.3,25,16.9,18.4,17.9,17,19.8,23.6,24.6,17.6,19.9,17.9,18.2,18.9,23.1,24.6,24.2,20,18.3,17.9]
X2 = [17.3,19.6,17.8,17.8,24.9,23.5,25.4,23.5,24.3,18.2,20.6,17.7,22.4,23.8,22.6,20.1,18.9,18.1,17.4,23.5,23.3,25.7,23.6,20.5,17.3,19.2,18.6,18.9,16.9,24.4,23.8,19.4,24.6,18.3,19.4,19,17.1,20.2,23.2,23.6,18.3,20.8,18,19.1,19.3,22.1,23.4,23.6,19.3,19.2,18.5]
Y = [67.03,66.17,68.39,66.42,74.40,75.95,77.21,83.18,81.89,75.29,71.46,77.58,76.04,74.46,68.06,70.24,68.60,76.05,67.30,77.23,83.12,77.20,76.74,73.73,63.03,65.67,62.36,68.82,65.54,82.37,80.72,74.54,74.48,69.92,75.79,73.41,70.39,78.93,73.67,77.03,77.64,72.26,68.10,75.88,77.50,74.11,74.86,75.77,68.00,77.72,70.47]

# Combine X1 and X2 into a single 2D array
X = np.column_stack((X1, X2))

# Create and fit the model
model = LinearRegression()
model.fit(X, Y)

# Predict Y values
Y_pred = model.predict(X)

# Plot the results
plt.scatter(X1, Y, color='blue', label='Actual')
plt.plot(X1, Y_pred, color='red', label='Predicted')
plt.xlabel('TEST SCORES')
plt.ylabel('HIGH SPEED INTERNET ACCESSIBILITY (%)')
plt.legend()
plt.title('Percentage of High Internet Speed Accessibility at Home vs ACT Test Scores')
plt.show()
