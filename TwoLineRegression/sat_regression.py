import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import pearsonr

"""
X1 - English
X2 - Math
Y - % High Speed Internet Access
"""

YEAR = 2019 # change year to change plot
if YEAR == 2023:
    # SAT 2023, HIGH SPEED
    X1 = [508, 512, 489, 495, 503, 494, 492, 489, 515, 493, 526, 458, 489, 497, 478]
    X2 = [488, 495, 469, 474, 463, 476, 478, 472, 493, 474, 508, 444, 468, 481, 445]
    Y = [75.95, 77.21, 83.18, 81.89, 75.29, 76.04, 74.46, 68.06, 83.12, 76.74, 82.37, 74.54, 77.03, 75.88, 68.00]

elif YEAR == 2021:
    # SAT 2021, HIGH SPEED
    X1 = [544, 545, 499, 500, 513, 502, 508, 523, 540, 526, 514, 505]
    X2 = [528, 527, 485, 487, 480, 483, 498, 508, 526, 531, 497, 498]
    Y = [72.31, 74.27, 75.80, 78.85, 69.08, 73.67, 76.23, 71.22, 81.13, 72.35, 76.70, 66.49]

elif YEAR == 2019:
    # SAT 2019, HIGH SPEED
    X1 = [518, 529, 499, 495, 516, 505, 509, 512, 507, 533, 503, 483]
    X2 = [506, 516, 486, 480, 483, 488, 504, 502, 496, 526, 492, 460]
    Y = [75.03, 70.68, 74.28, 79.50, 69.70, 71.92, 71.03, 69.38, 66.43, 79.13, 75.84, 63.68]


# DRIVER CODE
# regression lines
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
plt.xlim(math.floor(min(Y) / X_LIM_TO_MULT) * X_LIM_TO_MULT, min(100, math.ceil(max(Y) / X_LIM_TO_MULT) * X_LIM_TO_MULT))
plt.ylim(math.floor(min(X1 + X2) / Y_LIM_TO_MULT) * Y_LIM_TO_MULT, math.ceil(max(X1 + X2) / Y_LIM_TO_MULT) * Y_LIM_TO_MULT)  # % of internet access

# labels
plt.title(f"% of High Speed Internet Access vs. SAT Section Scores for 15 U.S. states ({YEAR})")
plt.xlabel("% of High Speed Internet Access")
plt.ylabel(f"Score on Specific SAT Section ({YEAR})")

# legend
math_legend = plt.legend(
    eng_plot, ["English SAT Scores", "Line of Best Fit (% High Speed Internet Access vs English SAT Scores)"],
    loc=1
)
eng_legend = plt.legend(
    math_plot, ["Math SAT Scores", "Line of Best Fit (% High Speed Internet Access vs Math SAT Scores)"],
    loc=2
)
corr_legend = plt.legend(
    [eng_plot[1], math_plot[1]], [f"r = {sat_eng_corr_coef:.3f}", f"r = {sat_math_corr_coef:.3f}"],
    loc=4
)
axis.add_artist(eng_legend)
axis.add_artist(math_legend)

plt.show()