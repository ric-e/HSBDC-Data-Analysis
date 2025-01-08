
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Sample data
'''
When running the code, comment everything else except a set each time (pair of X1, X2, Y) and save it accordingly
Make sure to comment and uncomment the labels that are not being used each time you run it
'''
X1 = [544,545,499,500,513,502,508,523,540,526,514,505]
X2 = [528,527,485,487,480,483,498,508,526,531,497]
Y = [72.31,74.27,75.80,78.85,69.08,73.67,76.23,71.22,81.13,67.17,72.35,76.70,66.49,75.88,68.00]

'''
#SAT 2023 
#HIGH SPEED
X1 = [592,553,596,610,546,508,512,489,495,503,539,565,494,492,489,610,626,616,611,551,515,560,493,601,601,603,607,631,591,526,538,458,522,570,652,525,486,574,547,489,527,605,606,497,621,563,569,549,478,615,604]
X2 = [570,529,587,582,536,488,495,469,474,463,515,549,476,478,482,598,619,592,583,529,493,551,474,599,583,588,586,621,576,508,528,444,516,557,634,519,468,551,531,468,501,602,585,481,618,536,544,532,445,621,596]
Y = [67.03,66.17,68.39,66.42,74.40,75.95,77.21,83.18,81.89,75.29,71.46,77.58,76.04,74.46,68.06,70.24,68.60,76.05,67.30,77.23,83.12,77.20,76.74,73.73,63.03,65.67,62.36,68.82,65.54,82.37,80.72,74.54,74.48,69.92,75.79,73.41,70.39,78.93,73.67,77.03,77.64,72.26,68.10,75.88,77.50,74.11,74.86,75.77,68.00,77.72,70.47]
#USAGE
X1 = [592,553,596,610,546,508,512,489,495,503,539,565,494,492,489,610,626,616,611,551,515,560,493,601,601,603,607,631,591,526,538,458,522,570,652,525,486,574,547,489,527,605,606,497,621,563,569,549,478,615,604]
X2 = [570,529,587,582,536,488,495,469,474,463,515,549,476,478,482,598,619,592,583,529,493,551,474,599,583,588,586,621,576,508,528,444,516,557,634,519,468,551,531,468,501,602,585,481,618,536,544,532,445,621,596]
Y = [78.85,73.56,77.62,75.85,79.01,81.58,76.87,83.49,87.88,74.86,81.69,79.37,86.85,83.10,77.93,77.76,78.02,81.84,75.95,85.87,84.33,79.07,81.22,83.28,75.33,77.21,71.14,74.22,72.34,82.69,81.02,78.61,74.90,69.99,82.93,79.12,80.63,86.88,75.18,79.56,80.84,80.73,73.22,79.72,79.85,76.46,80.04,80.85,77.74,87.88,83.07]
#SAT 2021
#HIGH SPEED
X1 = [591,567,592,610,527,544,545,499,500,513,551,572,502,508,551,623,619,616,605,558,542,591,523,626,612,614,618,625,596,540,562,508,526,578,631,525,535,565,566,514,529,605,618,505,621,571,584,537,520,604,626]
X2 = [568,553,589,584,530,528,527,485,487,480,534,572,483,498,544,620,623,603,583,541,531,593,508,636,589,606,607,620,598,526,563,488,531,571,628,523,507,554,557,497,507,610,602,498,617,553,567,535,487,611,607]
Y = [67.09,69.01,68.35,65.86,75.04,72.31,74.27,75.80,78.85,69.08,70.86,74.86,73.67,76.23,72.92,73.82,72.99,64.30,67.11,70.11,79.10,76.14,71.22,70.90,57.47,71.95,67.99,69.82,71.28,81.13,80.52,67.17,72.35,70.78,69.64,70.46,57.87,78.88,70.80,76.70,66.96,75.06,66.70,66.49,73.69,75.10,69.51,72.13,66.80,73.06,63.20]
#USAGE
X1 = [591,567,592,610,527,544,545,499,500,513,551,572,502,508,551,623,619,616,605,558,542,591,523,626,612,614,618,625,596,540,562,508,526,578,631,525,535,565,566,514,529,605,618,505,621,571,584,537,520,604,626]
X2 = [568,553,589,584,530,528,527,485,487,480,534,572,483,498,544,620,623,603,583,541,531,593,508,636,589,606,607,620,598,526,563,488,531,571,628,523,507,554,557,497,507,610,602,498,617,553,567,535,487,611,607]
Y = [71.66,78.85,74.40,75.56,76.41,80.10,71.88,75.99,80.68,69.35,75.76,74.92,85.60,84.28,78.56,79.81,79.51,75.60,76.12,80.94,77.87,76.44,77.65,82.05,71.91,77.98,76.50,76.42,67.01,81.01,80.78,73.13,72.36,73.22,73.34,76.25,71.50,84.12,71.20,77.55,74.80,81.10,74.28,71.82,79.85,79.04,74.83,79.29,75.30,82.83,72.64]
#SAT 2019 
#HIGH SPEED
X1 = [583,556,569,582,534,518,529,499,495,516,538,550,505,509,543,622,618,620,610,512,535,559,507,636,628,622,603,628,580,533,544,543,531,554,627,550,490,562,545,503,526,633,618,515,614,560,567,539,483,635,623]
X2 = [560,541,565,559,531,506,516,486,480,483,519,550,488,504,537,622,623,612,591,502,523,561,496,648,608,614,596,631,576,526,545,530,533,546,636,548,472,550,537,492,504,635,602,507,615,546,551,535,460,648,615]
Y = [63.16,66.42,71.83,62.71,69.00,75.03,70.68,74.28,79.50,69.70,63.19,77.86,71.92,71.03,67.36,65.80,65.25,58.20,59.70,69.38,78.87,72.01,66.43,72.50,53.91,63.65,64.31,69.81,70.01,79.13,74.32,64.16,69.94,63.14,69.29,67.32,62.64,73.85,68.17,75.84,71.39,70.41,63.21,67.63,75.63,73.07,70.97,68.60,63.68,67.77,60.39]
#USAGE
X1 = [583,556,569,582,534,518,529,499,495,516,538,550,505,509,543,622,618,620,610,512,535,559,507,636,628,622,603,628,580,533,544,543,531,554,627,550,490,562,545,503,526,633,618,515,614,560,567,539,483,635,623]
X2 = [560,541,565,559,531,506,516,486,480,483,519,550,488,504,537,622,623,612,591,502,523,561,496,648,608,614,596,631,576,526,545,530,533,546,636,548,472,550,537,492,504,635,602,507,615,546,551,535,460,648,615]
Y = [72.93,71.25,77.27,72.85,72.35,82.30,73.74,71.48,82.49,70.71,73.71,77.16,77.46,78.45,76.45,77.08,72.55,70.71,69.92,74.83,79.41,72.48,74.97,82.18,68.81,73.51,75.28,76.68,69.84,82.10,75.56,70.51,70.81,68.25,73.06,73.25,76.59,79.95,71.78,78.37,74.55,73.95,69.26,70.77,83.47,74.14,75.48,75.42,69.16,76.94,73.13]
'''
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
ax.set_zlabel('% of Internet Usage')
'''

ax.legend()
ax.set_title('% of High Speed Internet Access vs SAT Test Scores')
#ax.set_title('% of Internet Usage vs SAT Test Scores')

plt.show()
