# Error Excption 

import os

for i in range(10):

	try:
		z = 3/(i-3)
	except:
		continue
	else: 
		print(z)
	print(i)
