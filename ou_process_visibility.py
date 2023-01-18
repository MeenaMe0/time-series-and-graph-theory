import matplotlib.pyplot as plt
import numpy as np


# Set the number of time steps ,the noise and parameters
num_steps = 500 
dt = 0.01
speed = 10
sigma = 100
mean = 20
dwiener = np.random.normal(0, np.sqrt(dt),num_steps)    # difference of weiner process is normal distribution

# Generate the example of OU process and set the initial value
ou = []
ou.append(0)

for i in range (1,num_steps) :
    # Add the random value to list of random variable : ou
   ou.append(ou[i-1] + speed*(mean - ou[i-1])*dt + sigma*(dwiener[i-1]))

   # Check the previous value if it could see each other -> if it can then draw the connected line 
   
   # Check the previous value
   for j in range (0,i) :
        ch = 0
        # Check other value between them
        for k in range (j,i) :
            # Check if it can't draw the connected line
            if( ou[k] > ou[i] + (ou[j]-ou[i])*(i-k)/(i-j) ) :
                ch =1 
                break
        # If can then draw the connected line
        if (ch == 0) :
            plt.plot((i,j),(ou[i],ou[j])) # (x1 x2) , (y1,y2)

# Plot the ou model
plt.plot(ou , color = 'black')

# Draw the mean or expected value line : 
plt.plot((0,num_steps),(mean, mean) ,color = 'red')

plt.show()