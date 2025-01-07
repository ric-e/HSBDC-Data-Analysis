
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
Y = [72.93,71.25,77.27,72.85,72.35,82.30,73.74,71.48,82.49,70.71,73.71,77.16,77.46,78.45,76.45,77.08,72.55,70.71,69.92,74.83,79.41,72.48,74.97,82.18,68.81,73.51,75.28,76.68,69.84,82.10,75.56,70.51,70.81,68.25,73.06,73.25,76.59,79.95,71.78,78.37,74.55,73.95,69.26,70.77,83.47,74.14,75.48,75.42,69.16,76.94,73.13]

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
'''
ax.set_xlabel('English SAT Scores')
ax.set_ylabel('Math SAT Scores')
ax.set_zlabel('% of High Internet Access')
'''
ax.set_xlabel('English SAT Scores')
ax.set_ylabel('Math SAT Scores')
ax.set_zlabel('% of Internet Usage')


ax.legend()
#ax.set_title('% of High Speed Internet Access vs SAT Test Scores')
ax.set_title('% of Internet Usage vs SAT Test Scores')

plt.show()
