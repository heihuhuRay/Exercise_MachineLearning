# 20161024_
from __future__ import print_function

# Dictionary for the code
code = {'a':'n', 'b':'o', 'c':'p', 'd':'q', 'e':'r', 'f':'s', 'g':'t', 'h':'u', 
       'i':'v', 'j':'w', 'k':'x', 'l':'y', 'm':'z', 'n':'a', 'o':'b', 'p':'c', 
       'q':'d', 'r':'e', 's':'f', 't':'g', 'u':'h', 'v':'i', 'w':'j', 'x':'k',
       'y':'l', 'z':'m', 'A':'N', 'B':'O', 'C':'P', 'D':'Q', 'E':'R', 'F':'S', 
       'G':'T', 'H':'U', 'I':'V', 'J':'W', 'K':'X', 'L':'Y', 'M':'Z', 'N':'A', 
       'O':'B', 'P':'C', 'Q':'D', 'R':'E', 'S':'F', 'T':'G', 'U':'H', 'V':'I', 
       'W':'J', 'X':'K', 'Y':'L', 'Z':'M'}

letter_encoded = 'BZT! guvf fb obevat.' # Original message
letter_plainText = ''					# Decoded message, string
#letter_plainText = []
for Letter in letter_encoded:
	Decoded_Letter = code.get(Letter, Letter)    # dict.get(key, default=None)
	letter_plainText = letter_plainText + Decoded_Letter 
	#letter_plainText.append(Decoded_Letter)
print("The message is: ",letter_plainText)
#print(letter_plainText)

# dict.get(key, default=None)
''' dict = {'Name': 'Zara', 'Age': 27}
	print "Value : %s" %  dict.get('Age')
	print "Value : %s" %  dict.get('Sex', "Never")'''

import numpy as np

v = np.array( [1., 2., 3.] )
print(v)

A = np.array( [ [ 1, 2, 3, 4],
				[ 5, 6, 7, 8],
				[ 4, 3, 2, 1] ], dtype = np.uint8)
print(A)
print(A.shape)
print(A.reshape(4, 3))

matric_1_dimension = np.array( [1, 2, 3, 4, 5, 6, 7, 8] )
print(matric_1_dimension.shape)
print(matric_1_dimension.reshape(2,4))

A_Initialization = np.zeros((2, 3))
print(A_Initialization)
v_Initialization = np.ones(12)
print(v_Initialization)
print(np.random.uniform(1, 100, (10, 10)))
print(np.random.normal(0.5, 1.0, (10, 10)))
#print(np.random.uniform('a', 'z', (10, 10)))

matric_Manipulation = np.array(
						[
							[ 1, 2, 3, 4],
							[ 5, 6, 7, 8]
							])
print(matric_Manipulation[0, 2])
print(matric_Manipulation[:, 1])
print(matric_Manipulation[:, 1:3])

matric_Manipulation = np.array(
						[ 	[-1,  8,  8, -1],
							[ 1, -1, -1, -1] 
						]     )
negatives = matric_Manipulation < 0
print(negatives)
matric_Manipulation[negatives] = 0
print