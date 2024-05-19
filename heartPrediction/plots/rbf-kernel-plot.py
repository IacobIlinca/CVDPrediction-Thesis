import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC

# Generate a simple synthetic dataset that is clearly non-linear
X, y = make_circles(n_samples=100, factor=.1, noise=.1)

# Create an SVM classifier with RBF kernel
model = SVC(kernel='rbf', C=1.0)
model.fit(X, y)

# Create a mesh to plot
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Predict each point on the mesh
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(False)
plt.savefig("RBF_Kernel_SVM.png")
plt.show()
