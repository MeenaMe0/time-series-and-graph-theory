import matplotlib.pyplot as plt
import numpy as np


# Set the number of time steps ,the noise and parameters
num_steps = 500 
dt = 0.01
speed = 10
sigma = 3
mean = 20
dwiener = np.random.normal(0, np.sqrt(dt),num_steps)    # difference of weiner process is normal distribution

# Generate the example of cir process and set the initial value
cir = []
cir.append(0)

for i in range (1,num_steps) :
    # Add the random value to list of random variable : cir
   cir.append(cir[i-1] + speed*(mean - cir[i-1])*dt + sigma*(dwiener[i-1])*np.sqrt(np.abs(cir[i-1])))

# Plot the cir model
plt.plot(cir)

# Draw the mean or expected value line : 
plt.plot((0,num_steps),(mean, mean) ,color = 'grey')

plt.show()