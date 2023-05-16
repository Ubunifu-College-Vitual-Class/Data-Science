#Create a numpy array and plot it using matplotlib

#Solution 1
import numpy as np
import matplotlib.pyplot as plt 
x = np.arange(1, 11)
y = x * x 
plt.title("Line graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(x, y, color ="red")
plt.show()

#Solution 2
import numpy as np
import matplotlib.pyplot as plt 
x = np.arange(1, 11) #returns evenly spaced values within a given interval. #numpy.arange([start, ]stop, [step, ]dtype=None, *, like=None)
y = np.array([100, 10, 300, 20, 500, 60, 700, 80, 900, 100]) 
plt.title("Line graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(x, y, color ="green")
plt.show()
