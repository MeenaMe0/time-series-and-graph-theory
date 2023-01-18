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

    # Check the previous value
    for j in range (0,i) :
        ch = 0
        # Check other value between them
        for k in range (j,i) :
            # Check if it can't draw the connected line
            if( sde_ex[k] > sde_ex[i] + (sde_ex[j]-sde_ex[i])*(i-k)/(i-j) ) :
                ch =1 
                break
        # If can then draw the connected line
        if (ch == 0) :
            plt.plot((i,j),(sde_ex[i],sde_ex[j])) # (x1 x2) , (y1,y2)

# Plot the SDE model
plt.plot(sde_ex , color = 'black')

plt.show()