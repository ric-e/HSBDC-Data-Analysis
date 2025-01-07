import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Sample data
'''
When running the code, comment everything else except a set each time (pair of X1, X2, Y) and save it accordingly
Make sure to comment and uncomment the labels that are not being used each time you run it
'''
#SAT 2023 

#HIGH SPEED
X1 = [592,553,596,610,546,508,512,489,495,503,539,565,494,492,489,610,626,616,611,551,515,560,493,601,601,603,607,631,591,526,538,458,522,570,652,525,486,574,547,489,527,605,606,497,621,563,569,549,478,615,604]
X2 = [570,529,587,582,536,488,495,469,474,463,515,549,476,478,482,598,619,592,583,529,493,551,474,599,583,588,586,621,576,508,528,444,516,557,634,519,468,551,531,468,501,602,585,481,618,536,544,532,445,621,596]
Y = [67.03,66.17,68.39,66.42,74.40,75.95,77.21,83.18,81.89,75.29,71.46,77.58,76.04,74.46,68.06,70.24,68.60,76.05,67.30,77.23,83.12,77.20,76.74,73.73,63.03,65.67,62.36,68.82,65.54,82.37,80.72,74.54,74.48,69.92,75.79,73.41,70.39,78.93,73.67,77.03,77.64,72.26,68.10,75.88,77.50,74.11,74.86,75.77,68.00,77.72,70.47]


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
