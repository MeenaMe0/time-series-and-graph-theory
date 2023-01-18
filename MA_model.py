import matplotlib.pyplot as plt
import numpy as np


# Set the number of time steps and the white noise
num_steps = 500 
white_noise = np.random.randn(num_steps+2)      #normal random numstep+2 value

# Generate the ma model at thie T ,by adding white_noise from time T ,T-1 and T-2
ma_model = [0,0]
for i in range (2,num_steps) :
    # Add the random value to list of random variable : ma_model
   ma_model.append(white_noise[i-2]+white_noise[i-1]+white_noise[i])

# Plot the Binary process
plt.plot(ma_model)

# Draw the mean or expected value line : y = 0
plt.plot((0,num_steps),(0, 0) ,color = 'grey')

plt.show()
