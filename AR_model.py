import matplotlib.pyplot as plt
import numpy as np


# Set the number of time steps and the white noise
num_steps = 500 
white_noise = np.random.randn(num_steps)      #normal random numstep value

# Generate the ma model at thie T ,by adding value from time T-1 and T-2
ar_model = [0,0]
for i in range (2,num_steps) :
    # Add the random value to list of random variable : binary_p 
    # -0.5 * X_t-2 + 0.75 * X_t-1 + W_t
   ar_model.append(-0.5*ar_model[i-2]+0.75*ar_model[i-1]+white_noise[i])

# Plot the Binary process
plt.plot(ar_model)

# Draw the mean or expected value line : y = 0
plt.plot((0,num_steps),(0, 0) ,color = 'grey')

plt.show()