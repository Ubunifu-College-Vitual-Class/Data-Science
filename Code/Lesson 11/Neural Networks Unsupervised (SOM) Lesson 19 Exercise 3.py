# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score 

# Define the SOM class
class SOM:

    def __init__(self, m, n, dim):
    
    
        # Initialize the grid size and dimension
        self.m = m
        self.n = n
        self.dim = dim
        # Initialize the grid with random weights
        self.grid = np.random.rand(m, n, dim)
        # Initialize the learning rate and radius
        self.alpha = 0.8
        self.sigma = max(m, n) / 2.0
    
    
    def find_bmu(self, x):
    
    
        # Find the best matching unit for a given input x
        # Compute the Euclidean distance between x and each grid cell
        dist = np.linalg.norm(self.grid - x, axis=2)
        # Return the index of the cell with the minimum distance
        bmu = np.unravel_index(np.argmin(dist), (self.m, self.n))
        return bmu
    
    
    def update(self, x):    
    
        # Update the grid weights using a given input x
        # Find the best matching unit
        bmu = self.find_bmu(x)
        # Compute the distance between each cell and the bmu
        d = np.linalg.norm(np.dstack(np.mgrid[0:self.m, 0:self.n]) - bmu, axis=2)
        # Compute the neighborhood function
        h = np.exp(-d**2 / (2 * self.sigma**2))
        # Update the weights by moving them closer to x
        self.grid += self.alpha * h[:, :, np.newaxis] * (x - self.grid)
        # Decrease the learning rate and radius
        self.alpha *= 0.99
        self.sigma *= 0.99
    
    
    def train(self, data, epochs):    
    
        # Train the SOM using a given data set and number of epochs
        for i in range(epochs):
            # Shuffle the data
            np.random.shuffle(data)
        # For each input in the data set
        for x in data:
            # Update the grid weights
            self.update(x)





# Create a data set of random RGB colors
data = np.random.rand(2987, 3)

# Create a SOM object with a 20x30 grid and 3 dimensions
som = SOM(20, 30, 3)

# Train the SOM for 100 epochs
som.train(data, 100)

# Plot the SOM grid as an image
plt.imshow(som.grid)
plt.show()




