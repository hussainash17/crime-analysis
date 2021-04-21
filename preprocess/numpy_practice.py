import numpy as np

# creating an array using numpy
# create an even no of array
arr = np.arange(0, 10, 2)

# creating all zeros or ones of array
zeros = np.zeros((2,3))
ones = np.ones((3,2))

# create an array of 10 fives
five = np.ones(10)*5

# create an array of integers from 10 to 50
array_10to50 = np.arange(10, 50)

# create an array of even integers from 10 to 50
array_10to50 = np.arange(10, 50, 2)


# crating an evenly spaced point 1D array between a range
# creates 10 points between 0 to 5
arr = np.linspace(0, 5, 10)

# create an array of 20 linearly spaced points between 0 to 1
arr = np.linspace(0, 1, 20)

# create a 3x3 matrix with values ranging from 0 to 8
arr = np.arange(0, 9).reshape(3,3)

# create an identity matrix
arr = np.eye(3)

# create an arrays of random numbers in uniform distribution
# create an 1D array of random numbers between 0 to 1
arr = np.random.rand(5)

# create a random numbers between 0 to 1
arr = np.random.rand(1)

# 5x5 matrix of random number
arr = np.random.rand(5, 5)

# create an arrays of random numbers in standard normal distribution
# create an 1D array of random numbers
arr = np.random.randn(4)
arr = np.random.randn(4,4)

# random integers from low to high
# 10 random integers between 1 to 100
arr = np.random.randint(1, 100, 10)

arr = np.arange(25)
ranarr = np.random.randint(0, 50, 10)

# reshape 1D arr into different shape
arr = arr.reshape(5, 5)

# create a matrix
arr = (np.arange(1,101).reshape(10, 10)) / 100

# position of a max/min value in an array
arr = ranarr.argmax()

# indexing and selection
arr = np.arange(0, 11)
arr1 = arr[2:7]
arr = arr[:6]

# for 2D arrays
# make an array of 50 elements and reshape it to 5x10
arr = np.arange(50).reshape(5, 10)

# grab 24,25,26,27 and 34, 35, 36, 37
arr = arr[2:4, 4:8]

# conditional slicing
# take all the elements from arr where elements > 4
arr = np.arange(10)
arr = arr[arr > 4]
