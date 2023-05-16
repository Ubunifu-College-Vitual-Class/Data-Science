#Create a numpy array and plot it using matplotlib

#Solution 1 ; Create a 1D array of numbers from 0 to 4
import numpy as np
print(np.arange(10))

#Solution 2 ; Extract items that satisfy a given condition from 1D array
arr = np.arange(0,10)
arr[arr % 2 == 1] = -1
#print(arr)


#Solution 3 ; Reshape an array
arr = np.arange(10)
arr.reshape(2, -1)  # Setting to -1 automatically decides the number of cols
#print(arr)


#2b
#Solution 3 ; Create a 3x3x3 array with random values
arr = np.random.random((3,3,3))
#print(arr)

#Solution 4 ; 10x10 array with random values and find the minimum and maximum
arr = np.random.random((10,10))
Zmin, Zmax = arr.min(), arr.max()
#print(Zmin, Zmax)

#Solution 5 ; Get all the dates corresponding to the month of July
arr = np.arange('2023-07', '2023-08', dtype='datetime64[D]')
print(arr)
