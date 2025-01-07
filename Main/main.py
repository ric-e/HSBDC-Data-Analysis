import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Sample data
X1 = [17.4,19,16.8,18,25.8,24.4,26.7,25,26.6,18.5,20.7,16.6,22.5,24.5,22.1,19.7,18.3,18.1,17.8,25.1,24.5,26.4,24.3,19.3,17,19,17.4,18.3,16,25.1,24.5,19.3,25,16.9,18.4,17.9,17,19.8,23.6,24.6,17.6,19.9,17.9,18.2,18.9,23.1,24.6,24.2,20,18.3,17.9]
X2 = [17.3,19.6,17.8,17.8,24.9,23.5,25.4,23.5,24.3,18.2,20.6,17.7,22.4,23.8,22.6,20.1,18.9,18.1,17.4,23.5,23.3,25.7,23.6,20.5,17.3,19.2,18.6,18.9,16.9,24.4,23.8,19.4,24.6,18.3,19.4,19,17.1,20.2,23.2,23.6,18.3,20.8,18,19.1,19.3,22.1,23.4,23.6,19.3,19.2,18.5]
Y = [78.85,73.56,77.62,75.85,79.01,81.58,76.87,83.49,87.88,74.86,81.69,79.37,86.85,83.10,77.93,77.76,78.02,81.84,75.95,85.87,84.33,79.07,81.22,83.28,75.33,77.21,71.14,74.22,72.34,82.69,81.02,78.61,74.90,69.99,82.93,79.12,80.63,86.88,75.18,79.56,80.84,80.73,73.22,79.72,79.85,76.46,80.04,80.85,77.74,87.88,83.07]
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
