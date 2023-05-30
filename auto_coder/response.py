# The logic for generating values is wrong. It generates a random value between 0 and 5 and adds it to the previous y value. 
# We should get a random value between -5 and +5 and add it to the previous value to get a new y value.

import matplotlib.pyplot as plt
import numpy as np
import time

# create empty lists to store x and y values
x_values = []
y_values = []

for i in range(30):
    # generate a random number between -5 and +5 and add it to previous number
    y = np.random.uniform(-5, 5) + (y_values[-1] if y_values else 0)
    y_values.append(y)
    x_values.append(i)
    
    plt.plot(x_values, y_values)
    plt.draw()
    plt.pause(0.1)  # sleep for 0.1 seconds before updating the plot

plt.show()  # display the final plot