import numpy as np
import matplotlib.pyplot as plt

# Generate values for x from -10 to 10
x = np.linspace(-10, 10, 400)
# Calculate the sigmoid function
sigmoid = 1 / (1 + np.exp(-x))

# Plotting the sigmoid function
plt.figure(figsize=(8, 4))
plt.plot(x, sigmoid, label="Sigmoid Function")
plt.xlabel("x")
plt.ylabel("Sigmoid(x)")
plt.grid(True)
plt.legend()
plt.savefig("Sigmoid_Function_Plot.png")
plt.show()
