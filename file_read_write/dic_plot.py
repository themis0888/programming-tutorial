import collections, itertools
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os 
import operator

metadata_path = ''
tag_dict = {}
for i in range(6):
	tag_dict.update(dict(np.load(os.path.join(metadata_path, 'metadata_{}.npy'.format(i))).item()))

# dict: {tag:num_of_tag}
tag_num = dict(collections.Counter(itertools.chain.from_iterable(tag_dict.values())))

temp = {k:[str(v)] for k,v in tag_num.items()}

"""
temp = {}
for k, v in tag_num.items():
	if v > 100000:
		temp[k] = v
"""
num_num = dict(collections.Counter(itertools.chain.from_iterable(temp.values())))


result_fin = num_num
result_fin = [(elem1, elem2) for elem1, elem2 in result_fin]
result_fin = result_fin
zip(*result_fin)
font = {'size':5}
matplotlib.rc('font', **font)

plt.scatter(*zip(*result_fin), marker='.')
"""
lst = list(result_fin.keys())
plt.plot(lst, [result_fin[lst[i]] for i in range(len(lst))])
plt.ylabel('tag')
"""
plt.xticks(rotation=90)

i = 0
while os.path.isfile('dic_stat{0:3}.jpg'.format(i)):
	i += 1

plt.savefig('dic_stat{0:3}.jpg'.format(i))

