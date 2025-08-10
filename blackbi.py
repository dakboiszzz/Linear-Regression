# Libraries
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#  Implementing gradient descent

# Function to minimize
def f(x,y):
    func = x ** 2 + 2 * y ** 2 + x * y
    return func

# Computing the gradient
def grad(x,y):
    # Partial derivative with respect to x
    grad_x = 2 * x + y
    # Partial derivative with respect to y
    grad_y = 4 * y + x
    return [grad_x,grad_y]

# Gradient descent

# Initializing x and y
x, y = 2,2

# Learning rate
lr = 0.005

# Number of iterations
iterations = 200

# Bonus: Visualizing
path = []

# Implementing
for _ in range(iterations):
    x0,y0 = grad(x,y)
    x -= x0 * lr
    y -= y0 * lr
    path.append([x,y,f(x,y)])


# 3D Plot (Actually I have to ask AI for this part as I know nothing about graphing lol)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
path = np.array(path)
ax.plot3D(path[:, 0], path[:, 1], path[:, 2])
x_range = np.linspace(-3, 3, 100)
y_range = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = X**2 + 2*Y**2 + X*Y
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.5)
# Add regularized path
plt.show()
