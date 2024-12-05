import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

# from scipy import stats

# x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
# y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

# slope, intercept, r, p, std_err = stats.linregress(x, y)

# def myfunc(x):
#   return slope * x + intercept

# mymodel = list(map(myfunc, x))

# plt.scatter(x, y)
# plt.plot(x, mymodel)
# plt.show()


# # Units consumed and corresponding bill amounts
units = np.array([100, 200, 300, 400, 500, 600, 700, 800])  # Feature
bills = np.array([200, 400, 600, 800, 1000, 1200, 1400, 1600])  # Target

slope, intercept, r, p, std_err = stats.linregress(units, bills)


def myfunc(x):
  return slope * x + intercept

speed = myfunc(700)

print(speed)





