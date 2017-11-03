#!/usr/bin/env python
# coding=utf-8

# import numpy as np
# import matplotlib.pyplot as plt

# x = np.arange(0, 5, 0.1);
# y = np.sin(x)
# plt.plot(x, y)
# plt.show()

import numpy as np
import matplotlib.pyplot as mpl
from TFANN import MLPR
from sklearn.preprocessing import scale
 
pth = 'yahoostock.csv'
A = np.loadtxt(pth, delimiter=",", skiprows=1, usecols=(1, 4))
A = scale(A)
#y is the dependent variable
y = A[:, 1].reshape(-1, 1)
#A contains the independent variable
A = A[:, 0].reshape(-1, 1)


# Stock predictor
#Number of neurons in the input layer
i = 1
#Number of neurons in the output layer
o = 1
#Number of neurons in the hidden layers
h = 32
#The list of layer sizes
layers = [i, h, h, h, h, h, h, h, h, h, o]
mlpr = MLPR(layers, maxIter = 1000, tol = 0.70, reg = 0.001, verbose = True)

#Length of the hold-out period
nDays = 5
n = len(A)
#Learn the data
mlpr.fit(A[0:(n-nDays)], y[0:(n-nDays)])


#Begin prediction
yHat = mlpr.predict(A)
#Plot the results
mpl.plot(A, y, c='#b0403f')
mpl.plot(A, yHat, c='#5aa9ab')
mpl.show()
