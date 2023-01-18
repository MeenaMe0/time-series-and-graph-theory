import numpy as np
import matplotlib.pyplot as plt

# Set the number of time steps and the time step size
num_steps = 500
dt = 0.01
sigma = np.sqrt(dt)

# Initialize the Wiener process
# Make every element in list = 0
x = np.zeros(num_steps)

# Generate the Wiener process by summing up the increments
for i in range(1, num_steps):
    # value = previous value + random from normal distribution 
    # normal : mean = 0 , sd = sigma
    x[i] = x[i-1] + np.random.normal(0, sigma)

# Plot the Wiener process
plt.plot(x)
plt.show()
