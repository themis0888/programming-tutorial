a = 0

class hi:
	def __init__(self):
		self.aa = 11

	def bello(self, argss):
		print('Bello {}'.format(argss))

class hello(hi):
	def __init__(self, args):
		self.a = 1
		self.b = args 

	def hello(self, argss):
		print('Hello {}'.format(argss))
		print(self.a)
	print(a)

print(a)

a = hello(2)
print(a.a)
print(a.b)
a.hello(3)
a.bello(4)
