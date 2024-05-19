import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# Generate simple binary class data
np.random.seed(6)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
y = [0] * 20 + [1] * 20

# Create an SVM classifier with linear kernel
model = SVC(kernel='linear')
model.fit(X, y)

# Get the separating hyperplane
w = model.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (model.intercept_[0]) / w[1]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(xx, yy, 'k-', label="Hyperplane")
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', s=50, label="Data points")
plt.fill_between(xx, yy, 5, color='tab:orange', alpha=0.3)
plt.fill_between(xx, yy, -5, color='tab:blue', alpha=0.3)
plt.legend()
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.grid(True)
plt.savefig("Simple_Linear_SVM.png")
plt.show()
