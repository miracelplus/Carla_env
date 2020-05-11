import matplotlib
import matplotlib.pyplot as plt 
import numpy as np

def distance_reward(distance):
    assert distance >= 0
    if distance < 50:
        return (-5/2400)*distance**2 + 5/24
    else:
        return -5

a = np.linspace(0,100,101)
vecfunc = np.vectorize(distance_reward)
b = vecfunc(a)
plt.plot(b)
plt.show()