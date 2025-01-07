import numpy as np
from sklearn.linear_model import LinearRegression

# Example data
X = [
    [592,	570]
    [553,	529]
    [596,	587]
    [610,	582]
    [546,	536]
    [508,	488]
    [512,	495]
    [489,	469]
    [495,	474]
    [503,	463]
    [539,	515]
    [565,	549]
    [494,	476]
    [492,	478]
    [489,	482]
    [610,	598]
    [626,	619]
    [616,	592]
    [622,	583]
    [551,	529]
    [515,	493]
    [560,	551]
    [493,	474]
    [601,	599]
    [601,	583]
    [603,	588]
    [607,	586]
    [631,	621]
    [591,	576]
    [526,	508]
    [538,	528]
    [458,	444]
    [522,	516]
    [570,	557]
    [652,	634]
    [525,	519]
    [486,	468]
    [574,	551]
    [547,	531]
    [489,	468]
    [527,	501]
    [605,	602]
    [606,	585]
    [497,	481]
    [621,	618]
    [563,	536]
    [569,	544]
    [549,	532]
    [478,	445]
    [615,	621]
    [604,	596]
]

y = [570,529,587,582,536,488,495,469,474,463,515,549,476,478,482,598,619,592,583,529,493,551,474,599,583,588,586,621,576,508,528,444,516,557,634,519,468,551,531,468,501,602,585,481,618,536,544,532,445,621,596]

# Converting lists to numpy arrays
X_np = np.array(X)
y_np = np.array(y)

# Creating a linear regression model
model = LinearRegression()

# Fitting the model
model.fit(X_np, y_np)

# Making predictions
y_pred = model.predict(X_np)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Predictions:", y_pred)
