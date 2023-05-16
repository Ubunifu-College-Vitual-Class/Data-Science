#Create a 4X2 integer array and Prints its attributes
#Note: The element must be a type of unsigned int16. And print the following Attributes: –
#The shape of an array.
#Array dimensions.
#The Length of each element of the array in bytes.

import numpy

firstArray = numpy.empty([4,2], dtype = numpy.uint16) 
print("Printing Array")
print(firstArray)

print("Printing numpy array Attributes")
print("1> Array Shape is: ", firstArray.shape)
print("2>. Array dimensions are ", firstArray.ndim)
print("3>. Length of each element of array in bytes is ", firstArray.itemsize)
