import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import pearsonr

"""
X1 - English
X2 - Math
Y - % High Speed Internet Access
"""

'''
X1 = [82,90,84,93,85,84,85,92,90,87,86,85,94,83,86,79,89,77,83,88,84,86,79,75,82,80,77,76,85,88,86,77,86,96,77,78,87,79,81]  #English 
X2 = [34,70,46,69,56,55,53,70,65,52,49,53,70,47,26,49,55,40,38,57,43,52,28,36,51,49,45,41,52,58,57,47,58,70,47,44,55,51,43]  #Math
Y = [84.5,92,85.5,92,90.5,90,91,91.5,90,90,89.5,89,92,83,82,84.5,90,79,77,89.5,81,87,73,80,84,86,84,80.5,91,86.5,88.5,87,86.5,91.5,85.5,81.5,81,88,85] #Internet High Speed Accessibility
'''



YEAR = (2023, "EQAO") # change year to change plot
if YEAR[0] == 2023 and YEAR[1] == "SAT":
    # SAT 2023, HIGH SPEED
    X1 = [508, 512, 489, 495, 503, 494, 492, 489, 493, 458, 489, 478, 515, 497, 526]
    X2 = [488, 495, 469, 474, 463, 476, 478, 472, 474, 444, 468, 445, 493, 481, 508]
    Y = [75.95, 77.21, 83.18, 81.89, 75.29, 76.04, 74.46, 68.06, 76.74, 74.54, 77.03, 68.00, 83.12, 75.88, 82.37]

elif YEAR[0] == 2021 and YEAR[1] == "SAT":
    # SAT 2021, HIGH SPEED
    X1 = [544, 545, 499, 500, 513, 502, 508, 523, 540, 526, 514, 505]
    X2 = [528, 527, 485, 487, 480, 483, 498, 508, 526, 531, 497, 498]
    Y = [72.31, 74.27, 75.80, 78.85, 69.08, 73.67, 76.23, 71.22, 81.13, 72.35, 76.70, 66.49]

elif YEAR[0] == 2019 and YEAR[1] == "SAT":
    # SAT 2019, HIGH SPEED
    X1 = [518, 529, 499, 495, 516, 505, 509, 512, 507, 533, 503, 483]
    X2 = [506, 516, 486, 480, 483, 488, 504, 502, 496, 526, 492, 460]
    Y = [75.03, 70.68, 74.28, 79.50, 69.70, 71.92, 71.03, 69.38, 66.43, 79.13, 75.84, 63.68]

elif YEAR[0] == 2023 and YEAR[1] == "ACT":
    # ACT 2023, 50 STATES, HIGH SPEED
    #X1 = [17.4,19,16.8,18,25.8,24.4,26.7,25,26.6,18.5,20.7,16.6,22.5,24.5,22.1,19.7,18.3,18.1,17.8,25.1,24.5,26.4,24.3,19.3,17,19,17.4,18.3,16,25.1,24.5,19.3,25,16.9,18.4,17.9,17,19.8,23.6,24.6,17.6,19.9,17.9,18.2,18.9,23.1,24.6,24.2,20,18.3,17.9]
    X1 = [17.4, 18, 18.1, 17.8, 17, 17.4, 18.3, 16, 16.9, 17.9, 17, 17.9, 18.9, 18.3, 17.9]
    #X2 = [17.3,19.6,17.8,17.8,24.9,23.5,25.4,23.5,24.3,18.2,20.6,17.7,22.4,23.8,22.6,20.1,18.9,18.1,17.4,23.5,23.3,25.7,23.6,20.5,17.3,19.2,18.6,18.9,16.9,24.4,23.8,19.4,24.6,18.3,19.4,19,17.1,20.2,23.2,23.6,18.3,20.8,18,19.1,19.3,22.1,23.4,23.6,19.3,19.2,18.5]
    X2 = [17.3, 17.8, 18.1, 17.4, 17.3, 18.6, 18.9, 16.9, 18.3, 19, 17.1, 18, 19.3, 19.2, 18.5]
    #Y = [67.03,66.17,68.39,66.42,74.40,75.95,77.21,83.18,81.89,75.29,71.46,77.58,76.04,74.46,68.06,70.24,68.60,76.05,67.30,77.23,83.12,77.20,76.74,73.73,63.03,65.67,62.36,68.82,65.54,82.37,80.72,74.54,74.48,69.92,75.79,73.41,70.39,78.93,73.67,77.03,77.64,72.26,68.10,75.88,77.50,74.11,74.86,75.77,68.00,77.72,70.47]  
    Y = [67.03, 66.42, 76.05, 67.30, 63.03, 62.36, 68.82, 65.54, 69.92, 73.41, 70.39, 68.10, 77.50, 77.72, 70.47]

elif YEAR[0] == 2023 and YEAR[1] == "EQAO":
    # EQAO 2023, ONTARIO, HIGH SPEED
    X1 = [82,90,84,93,85,84,85,92,90,87,86,85,94,83,86,79,89,77,83,88,84,86,79,75,82,80,77,76,85,88,86,77,86,96,77,78,87,79,81]
    X2 = [34,70,46,69,56,55,53,70,65,52,49,53,70,47,26,49,55,40,38,57,43,52,28,36,51,49,45,41,52,58,57,47,58,70,47,44,55,51,43]
    Y = [84.5,92,85.5,92,90.5,90,91,91.5,90,90,89.5,89,92,83,82,84.5,90,79,77,89.5,81,87,73,80,84,86,84,80.5,91,86.5,88.5,87,86.5,91.5,85.5,81.5,81,88,85]

assert len(X1) == len(Y) and len(X2) == len(Y) # make sure 1-1-1 correspondence

# DRIVER CODE
# regression lines - adapted from https://stackoverflow.com/questions/6148207/linear-regression-with-matplotlib-numpy
sat_eng_coef = np.polyfit(Y, X1, 1)
sat_eng_1d_fn = np.poly1d(sat_eng_coef)
sat_math_coef = np.polyfit(Y, X2, 1)
sat_math_1d_fn = np.poly1d(sat_math_coef)

# correlation coefficients
sat_eng_corr_coef, _ = pearsonr(Y, X1)
sat_math_corr_coef, _ = pearsonr(Y, X2)

# plotting
axis = plt.subplot()

eng_plot = axis.plot(Y, X1, "yo", Y, sat_eng_1d_fn(Y), "k-")
math_plot = axis.plot(Y, X2, "bo", Y, sat_math_1d_fn(Y), "g-")

# axis limits
X_LIM_TO_MULT = 10
Y_LIM_TO_MULT = 50

XLIM_MIN = math.floor(min(Y) / X_LIM_TO_MULT) * X_LIM_TO_MULT
XLIM_MAX = min(100, math.ceil(max(Y) / X_LIM_TO_MULT) * X_LIM_TO_MULT)

YLIM_MIN = math.floor(min(X1 + X2) / Y_LIM_TO_MULT) * Y_LIM_TO_MULT
YLIM_MAX = math.ceil(max(X1 + X2) / Y_LIM_TO_MULT) * Y_LIM_TO_MULT
YLIM_MAX = min(36, YLIM_MAX) if YEAR[1] == "ACT" else YLIM_MAX

plt.xlim(XLIM_MIN, XLIM_MAX)
plt.ylim(YLIM_MIN, YLIM_MAX)  # % of internet access

# labels
TITLE = f"% of students meeting Ontario provincial standards ({YEAR[0]})" if YEAR[1] == "EQAO" else f"{YEAR[1]} Section Scores for {min(len(Y), 50)} U.S. states ({YEAR[0]})"
TITLE += " vs. % of High Speed Internet Access"

YLABEL = f"% of students meeting Ontario provincial standards" if YEAR[1] == "EQAO" else f"Score on Specific {YEAR[1]} Section ({YEAR[0]})"

plt.title(TITLE)
plt.ylabel(YLABEL)
plt.xlabel("% of High Speed Internet Access")

# legend
math_legend = plt.legend(
    eng_plot, [f"English {YEAR[1]} Scores", f"Line of Best Fit (% High Speed Internet Access vs English {YEAR[1]} Scores)"],
    loc=1
)
eng_legend = plt.legend(
    math_plot, [f"Math {YEAR[1]} Scores", f"Line of Best Fit (% High Speed Internet Access vs Math {YEAR[1]} Scores)"],
    loc=2
)
corr_legend = plt.legend(
    [eng_plot[1], math_plot[1]], [f"r = {sat_eng_corr_coef:.3f}", f"r = {sat_math_corr_coef:.3f}"],
    loc=4
)
axis.add_artist(eng_legend)
axis.add_artist(math_legend)

plt.show()
