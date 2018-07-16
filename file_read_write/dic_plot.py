import collections, itertools
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os 
import operator


a_dic = dict(np.load('./metadata_0.npy').item())



#a_dic = {'file1':["a","b","c"], 'file2':["b","c","d"], 'file3':["c","d","e"]}
tag_num = dict(collections.Counter(itertools.chain.from_iterable(a_dic.values())))

#temp = {k:[str(v)] for k,v in tag_num.items()}

temp = {}
for k, v in tag_num.items():
	if v > 20000:
		temp[k] = v

# num_num = dict(collections.Counter(itertools.chain.from_iterable(temp.values())))
"""
b_dic = {'file1':["a","b","c"], 'file2':["b","c","d"], 'file3':["c","d","e"]}
result2 = dict(collections.Counter(itertools.chain.from_iterable(b_dic.values())))
"""

"""
result_fin = {}
for i in result.keys():
	if i in result:
		if result[i] > 0:
			if i not in result_fin: 
				result_fin[i]=0
			result_fin[i]+=result[i]

for i in result2.keys():
	if i not in result_fin: 
		result_fin[i]=0
	if i in result2:
		result_fin[i]+=result2[i]
"""
#result_fin = {int(k):v for k,v in temp.items()}
#result_fin = collections.OrderedDict(sorted(result.items()))
result = temp
# result_fin = collections.OrderedDict(sorted(result_fin.items()))
result_fin = sorted(result.items(), key=operator.itemgetter(1))
# result_fin = [(elem1, np.log(elem2)) for elem1, elem2 in result_fin]
zip(*result_fin)

fig, axs = plt.subplots(1,1)

axs.scatter(*zip(*result_fin), '.')
axs.set_yscale('log')
axs.set_ylim([0, np.maximum(np.log(result_fin))])


"""
lst = list(result_fin.keys())
plt.plot(lst, [result_fin[lst[i]] for i in range(len(lst))])
plt.ylabel('tag')
"""
i = 0
while os.path.isfile('dic_stat{0:3}.jpg'.format(i)):
	i += 1

plt.savefig('dic_stat{0:3}.jpg'.format(i))

