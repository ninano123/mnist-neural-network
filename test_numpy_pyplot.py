import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0,12,0.01)

plt.figure(figsize=(10,6))
plt.plot(t,np.sin(t))
plt.plot(t,np.cos(t))
plt.show()

a = np.zeros([3,2])
print(a)
a[0]=[0,1]
a[1]=[3,1]
a[2]=[0,12]
print(a)

plt.imshow(a, interpolation = "nearest")
plt.show()
