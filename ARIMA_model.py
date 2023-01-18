import matplotlib.pyplot as plt
import numpy as np


# Set the number of time steps and the white noise
num_steps = 500 
white_noise = np.random.randn(num_steps+2)      #normal random numsteps+2 value

# Generate the arma model at thie T ,by adding value from time T-1, T-2 and white_noise from time T ,T-1 , T-2
arma_model = [0,0]
for i in range (2,num_steps) :
    # Add the random value to list of random variable : arma_model
    # -0.5 * X_t-2 + 0.75 * X_t-1 + W_t + W_t-1 + W_t-2
   arma_model.append(-0.5*arma_model[i-2]+0.75*arma_model[i-1]+white_noise[i]+white_noise[i-1]+white_noise[i-2])

# Plot the arma model
plt.subplots()
plt.plot(arma_model)
plt.title("After 1 Difference")

# Draw the mean or expected value line : y = 0
plt.plot((0,num_steps),(0, 0) ,color = 'grey')

# Generate model that the difference is arma model
arima_model= [0]
mean = 0

for i in range (1,num_steps) :
    arima_model.append(arima_model[i-1] + arma_model[i])
    mean +=arima_model[i]

# Plot the arima model
plt.subplots()
plt.plot(arima_model)
plt.title("Before 1 Difference")

# Draw the mean or expected value line : y = 0
mean= mean /num_steps
plt.plot((0,num_steps),(mean, mean) ,color = 'grey')

plt.show()