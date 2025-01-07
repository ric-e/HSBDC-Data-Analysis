import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Sample data
X1 = [18.7,18.9,17.9,19,22.4,23.8,25.9,23.9,23.6,19.5,20.9,17.9,21.9,24.5,21.7,20.8,20.5,19.5,18.7,24.2,22,25.5,24.5,20.3,18.1,20.3,18.7,19.4,16.8,24.9,24.3,18.1,24.1,17.8,18.6,19,18.2,20.3,23.3,25,18,20.7,19.1,19.5,19.5,23.9,23.8,21.3,20.6,19.4,18.8]
X2 = [18.1,19.9,19.2,18.7,22.3,23.3,24.7,23.1,22.7,19.5,20.7,19.2,21.9,23.8,22.3,21,20.7,19.2,18.2,23.8,21.7,25.2,24,21.4,17.9,20.2,19.7,19.7,18,24.7,24,18.9,24.1,19.2,19.9,19.9,18.3,20.8,23.1,23.9,18.7,21.3,18.9,20.4,20,23.3,23.3,22.1,19.7,20.2,19.4]
Y = [63.16,66.42,71.83,62.71,69.00,75.03,70.68,74.28,79.50,69.70,63.19,77.86,71.92,71.03,67.36,65.80,65.25,58.20,59.70,69.38,78.87,72.01,66.43,72.50,53.91,63.65,64.31,69.81,70.01,79.13,74.32,64.16,69.94,63.14,69.29,67.32,62.64,73.85,68.17,75.84,71.39,70.41,63.21,67.63,75.63,73.07,70.97,68.60,63.68,67.77,60.39]
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
