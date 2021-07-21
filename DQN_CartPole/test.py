import numpy as np

states = [1,2,3,4]

array_dummy = np.array(states)
array_dummy[len(array_dummy)-1] = 7
#array_dummy = array_dummy.reshape(4,1)


print(array_dummy.shape)