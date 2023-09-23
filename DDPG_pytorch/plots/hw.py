import numpy as np

x = list(map(float, input("Enter x: ").strip().split()))[:2]

x = np.array(x)
w = np.array([1, 1])
net = w@x
out = net >= 1.5
out = int(out)
result = {0: "Healthy", 1: "Disease"}

print("Net = %.2f" % net, "\tResult: ", result[out])