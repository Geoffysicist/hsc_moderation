import numpy as np
n = 5

that_array = np.rint(np.empty(n))

for ctr in range(n+3):
    this_array = np.random.randint(10, size=n)
    
    if ctr:
        that_array = np.append(that_array, [this_array], axis = 0)
    else:
        that_array = np.append([that_array], [this_array], axis = 0)

    

# print(that_array)
that_array = np.delete(that_array,0,0)
print(that_array)
print(np.sum(that_array,axis=0))
print(np.std(that_array, axis=0, ddof=1))
print(np.std(that_array, axis=0))
print(np.arange(1,n+1))
    

# a = np.array([1,2,3])
# b = np.array([2,4,6])

# print(a, b)
# c = np.append([a],[b], axis=0)
# print(c)
# d = np.append(c,[a], axis=0)
# print(d)