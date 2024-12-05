
import matplotlib.pyplot as plt
import numpy
from sklearn.linear_model import LinearRegression
from scipy import stats


age = [2, 2, 3, 5, 6, 7, 10,15,20,25,30,35,40,50,60,70,80,90]
weight = [1, 2, 2, 4, 8, 10, 15,20,40,50,50,50,45,40,30,20,10,5]



mymodel = numpy.poly1d(numpy.polyfit(age, weight, 3))

print(mymodel(46))


# mymodel = np.poly1d(np.polyfit(x, y, 3))
# myline = np.linspace(1, 90, 100)

# plt.scatter(x, y)
# plt.plot(myline, mymodel(myline))
# plt.show()


# POC
