import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Sample data
X1 = [593,547,563,614,531,606,530,503,482,520,535,544,513,559,542,641,632,631,611,513,536,555,509,644,634,640,605,629,563,532,530,577,528,546,635,578,530,560,540,539,543,612,623,513,624,562,561,541,558,642,626]
X2 = [572,533,553,594,524,595,512,492,468,497,515,541,493,556,532,635,628,616,586,499,524,551,495,651,607,631,591,625,553,520,526,561,523,535,621,570,517,548,531,524,521,603,604,507,614,551,541,534,528,649,604]
Y = [55.97,66.25,65.78,57.13,69.16,59.12,70.68,68.78,72.46,70.74,68.84,71.53,68.03,71.98,62.75,67.82,63.75,61.82,59.43,71.91,72.40,68.93,62.52,69.65,52.58,64.09,59.79,67.72,72.12,78.21,73.52,57.78,65.69,65.27,68.34,64.52,59.84,72.65,66.01,75.39,60.50,61.91,62.26,65.22,75.62,71.01,73.57,76.21,60.98,69.72,65.88]
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

ax.set_xlabel('English SAT Scores')
ax.set_ylabel('Math SAT Scores')
ax.set_zlabel('% of High Internet Access')
ax.legend()
ax.set_title('% of High Internet Access vs SAT Test Scores')

plt.show()
