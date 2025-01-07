import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Sample data
X1 = [18.4,19.4,18.6,18.6,26.2,23.3,27.5,25.9,25.8,19.9,22.1,16.8,22.3,25.3,22.2,20.4,18.9,18.7,18.1,25.7,25.7,27.6,25.1,20.2,17.5,19.9,19.2,19.1,16.7,26.5,25.3,19.7,26.1,17.3,18.2,18.5,19.1,19.6,24.8,25.7,17.4,20.5,18.7,18.9,19.7,24.2,25.5,22.9,20.6,18.9,18.6]
X2 = [18,20.5,19.8,18.3,25.6,23,26.2,24.5,24.5,19.6,21.9,18.1,22.3,24.5,22.8,20.8,19.5,18.8,17.8,24.4,24.5,26.9,24.5,21.5,17.6,19.9,20,19.6,17.7,25.9,24.7,20.1,25.7,19,19.7,19.5,18.7,20.3,24.3,24.8,18.4,21.2,18.5,20,20.1,23.4,24.5,23.1,19.6,19.9,19.4]
Y = [71.66,78.85,74.40,75.56,76.41,80.10,71.88,75.99,80.68,69.35,75.76,74.92,85.60,84.28,78.56,79.81,79.51,75.60,76.12,80.94,77.87,76.44,77.65,82.05,71.91,77.98,76.50,76.42,67.01,81.01,80.78,73.13,72.36,73.22,73.34,76.25,71.50,84.12,71.20,77.55,74.80,81.10,74.28,71.82,79.85,79.04,74.83,79.29,75.30,82.83,72.64]
X = np.column_stack((X1, X2))

# Create and fit the model
model = LinearRegression()
model.fit(X, Y)

# Predict Y values
Y_pred = model.predict(X)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the actual data points
ax.scatter(X1, X2, Y, color='blue', label='Actual')

# Plot the regression plane
X1_grid, X2_grid = np.meshgrid(np.linspace(min(X1), max(X1), 10), np.linspace(min(X2), max(X2), 10))
Y_grid = model.predict(np.column_stack((X1_grid.ravel(), X2_grid.ravel()))).reshape(X1_grid.shape)
ax.plot_surface(X1_grid, X2_grid, Y_grid, color='red', alpha=0.5, label='Regression Plane')

ax.set_xlabel('English ACT Scores')
ax.set_ylabel('Math ACT Scores')
ax.set_zlabel('% of Total Internet Usage at Home')
ax.legend()
ax.set_title('% of Total Internet Usage vs ACT Test Scores')

plt.show()
