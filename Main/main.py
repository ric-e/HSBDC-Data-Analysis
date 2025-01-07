import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Sample data
X1 = [18.7,18.9,17.9,19,22.4,23.8,25.9,23.9,23.6,19.5,20.9,17.9,21.9,24.5,21.7,20.8,20.5,19.5,18.7,24.2,22,25.5,24.5,20.3,18.1,20.3,18.7,19.4,16.8,24.9,24.3,18.1,24.1,17.8,18.6,19,18.2,20.3,23.3,25,18,20.7,19.1,19.5,19.5,23.9,23.8,21.3,20.6,19.4,18.8]
X2 = [18.1,19.9,19.2,18.7,22.3,23.3,24.7,23.1,22.7,19.5,20.7,19.2,21.9,23.8,22.3,21,20.7,19.2,18.2,23.8,21.7,25.2,24,21.4,17.9,20.2,19.7,19.7,18,24.7,24,18.9,24.1,19.2,19.9,19.9,18.3,20.8,23.1,23.9,18.7,21.3,18.9,20.4,20,23.3,23.3,22.1,19.7,20.2,19.4]
Y = [72.93,71.25,77.27,72.85,72.35,82.30,73.74,71.48,82.49,70.71,73.71,77.16,77.46,78.45,76.45,77.08,72.55,70.71,69.92,74.83,79.41,72.48,74.97,82.18,68.81,73.51,75.28,76.68,69.84,82.10,75.56,70.51,70.81,68.25,73.06,73.25,76.59,79.95,71.78,78.37,74.55,73.95,69.26,70.77,83.47,74.14,75.48,75.42,69.16,76.94,73.13]

# Combine X1 and X2 into a single 2D array
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
