#MyScript.py
#Implements the Hello World
import numpy as np
text = 'Hello World'
print(text)
'''
list_words = ['Mary','had','a','little','lamb']
for idx in range(len(list_words)):
	print(idx, list_words[idx])

list_words = ['Mary','had','a','little','lamb']
for idx, word in enumerate(list_words):
	print(idx, word)
'''
vector = np.array( [[1, 2, 3],
				   [1.2, 30.3, 22],
				   [33, 11, 44] ])#every elements in np array must be same
print(vector)
print("vector_size =", vector.size)
print("vector_shape =", vector.shape)
print("vector_dimensions =", vector.ndim)
print("vector_dtype =", vector.dtype.name)
print("vector_itemsize =", vector.itemsize)
print(vector[1][0])
print("vector_data =", vector.data)
print(type(vector))