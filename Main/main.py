import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Sample data
'''
When running the code, comment everything else except a set each time (pair of X1, X2, Y) and save it accordingly
Make sure to comment and uncomment the labels that are not being used each time you run it
'''
X1 = [591,567,592,610,527,544,545,499,500,513,551,572,502,508,551,623,619,616,605,558,542,591,523,626,612,614,618,625,596,540,562,508,526,578,631,525,535,565,566,514,529,605,618,505,621,571,584,537,520,604,626]
X2 = [568,553,589,584,530,528,527,485,487,480,534,572,483,498,544,620,623,603,583,541,531,593,508,636,589,606,607,620,598,526,563,488,531,571,628,523,507,554,557,497,507,610,602,498,617,553,567,535,487,611,607]
Y = [67.09,69.01,68.35,65.86,75.04,72.31,74.27,75.80,78.85,69.08,70.86,74.86,73.67,76.23,72.92,73.82,72.99,64.30,67.11,70.11,79.10,76.14,71.22,70.90,57.47,71.95,67.99,69.82,71.28,81.13,80.52,67.17,72.35,70.78,69.64,70.46,57.87,78.88,70.80,76.70,66.96,75.06,66.70,66.49,73.69,75.10,69.51,72.13,66.80,73.06,63.20]

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
'''
ax.set_xlabel('English SAT Scores')
ax.set_ylabel('Math SAT Scores')
ax.set_zlabel('% of Internet Accessibility Access')
'''

ax.legend()
ax.set_title('% of High Speed Internet Access vs SAT Test Scores')
#ax.set_title('% of Internet Usage vs SAT Test Scores')

plt.show()
