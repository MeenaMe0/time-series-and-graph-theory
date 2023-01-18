import matplotlib.pyplot as plt
import numpy as np


# Set the number of time steps ,the noise and parameters
num_steps = 500 
dt = 0.01
speed = 10
sigma = 3
dwiener = np.random.normal(0, np.sqrt(dt),num_steps)    # difference of weiner process is normal distribution

# Generate the example of SDE and set the initial value
sde_ex = []
sde_ex.append(0)

for i in range (1,num_steps) :
    # Add the random value to list of random variable : sde_ex
   sde_ex.append(sde_ex[i-1] - speed*(sde_ex[i-1])*dt + sigma*(dwiener[i-1]))

# Plot the SDE model
plt.plot(sde_ex)
plt.show()