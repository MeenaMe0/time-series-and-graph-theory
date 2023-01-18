import matplotlib.pyplot as plt
import numpy as np


# Set the number of time steps and the time step size
num_steps = 500 

# Set the probability : random value from a prob list 
prob = [-1,1,1,1,1] # p = 0.8
#prob = [-1,1] # p= 0.5 

# Generate the Random walk by random for num_steps times
random_w = [0]
for i in range (1,num_steps) :
    # Add the random value to list of random variable : random_w
    random_w.append(random_w[i-1] + np.random.choice(prob))

# Plot the Random walk
plt.plot(random_w)

# Draw the mean or expected value line : 𝑎+𝑡(2𝑝−1)
startline = random_w[0] # a
mean = 2*0.8 -1 # 2p -1
plt.plot((0,num_steps),(startline,startline + num_steps*(mean)) ,color = 'grey')

plt.show()
