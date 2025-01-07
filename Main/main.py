import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Sample data
X1 = [18.4,19.4,18.6,18.6,26.2,23.3,27.5,25.9,25.8,19.9,22.1,16.8,22.3,25.3,22.2,20.4,18.9,18.7,18.1,25.7,25.7,27.6,25.1,20.2,17.5,19.9,19.2,19.1,16.7,26.5,25.3,19.7,26.1,17.3,18.2,18.5,19.1,19.6,24.8,25.7,17.4,20.5,18.7,18.9,19.7,24.2,25.5,22.9,20.6,18.9,18.6]
X2 = [18,20.5,19.8,18.3,25.6,23,26.2,24.5,24.5,19.6,21.9,18.1,22.3,24.5,22.8,20.8,19.5,18.8,17.8,24.4,24.5,26.9,24.5,21.5,17.6,19.9,20,19.6,17.7,25.9,24.7,20.1,25.7,19,19.7,19.5,18.7,20.3,24.3,24.8,18.4,21.2,18.5,20,20.1,23.4,24.5,23.1,19.6,19.9,19.4]
Y = [67.09,69.01,68.35,65.86,75.04,72.31,74.27,75.80,78.85,69.08,70.86,74.86,73.67,76.23,72.92,73.82,72.99,64.30,67.11,70.11,79.10,76.14,71.22,70.90,57.47,71.95,67.99,69.82,71.28,81.13,80.52,67.17,72.35,70.78,69.64,70.46,57.87,78.88,70.80,76.70,66.96,75.06,66.70,66.49,73.69,75.10,69.51,72.13,66.80,73.06,63.20]
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
ax.set_zlabel('% of High Speed Internet Access')
ax.legend()
ax.set_title('% of High Speed Internet Access vs ACT Test Scores')

plt.show()
