import matplotlib.pyplot as plt
import numpy as np


# Set the number of time steps and the time step size
num_steps = 500 

# Set the probability : random value from a prob list 
prob = [-1,1,1,1,1] # p = 0.8
#prob = [-1,1] # p= 0.5 

# Generate the Binary process by random for num_steps times
binary_p = []
for i in range (0,num_steps) :
    # Add the random value to list of random variable : binary_p 
    binary_p.append(np.random.choice(prob))

# Plot the Binary process
plt.plot(binary_p)

# Draw the mean or expected value line : 2*p = 1
mean = 2*0.8 -1
plt.plot((0,num_steps),(mean,mean) ,color = 'grey')

# Set the graph frame range -2 to 2
plt.ylim([-2,2])
plt.show()
