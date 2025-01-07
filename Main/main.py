import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Sample data
'''
When running the code, comment everything else except a set each time (pair of X1, X2, Y) and save it accordingly
Make sure to comment and uncomment the labels that are not being used each time you run it
'''
X1 = [583,556,569,582,534,518,529,499,495,516,538,550,505,509,543,622,618,620,610,512,535,559,507,636,628,622,603,628,580,533,544,543,531,554,627,550,490,562,545,503,526,633,618,515,614,560,567,539,483,635,623]
X2 = [560,541,565,559,531,506,516,486,480,483,519,550,488,504,537,622,623,612,591,502,523,561,496,648,608,614,596,631,576,526,545,530,533,546,636,548,472,550,537,492,504,635,602,507,615,546,551,535,460,648,615]
Y = [63.16,66.42,71.83,62.71,69.00,75.03,70.68,74.28,79.50,69.70,63.19,77.86,71.92,71.03,67.36,65.80,65.25,58.20,59.70,69.38,78.87,72.01,66.43,72.50,53.91,63.65,64.31,69.81,70.01,79.13,74.32,64.16,69.94,63.14,69.29,67.32,62.64,73.85,68.17,75.84,71.39,70.41,63.21,67.63,75.63,73.07,70.97,68.60,63.68,67.77,60.39]





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
