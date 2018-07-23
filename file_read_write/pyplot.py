import collections, itertools
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


a_dic = {'file1':["a","b","c"], 'file2':["b","c","d"], 'file3':["c","d","e"]}
result = dict(collections.Counter(itertools.chain.from_iterable(a_dic.values())))

b_dic = {'file1':["a","b","c"], 'file2':["b","c","d"], 'file3':["c","d","e"]}
result2 = dict(collections.Counter(itertools.chain.from_iterable(b_dic.values())))



result_fin = {}
for i in result.keys():
	if i not in result_fin: 
		result_fin[i]=0
	if i in result:
		result_fin[i]+=result[i]

for i in result2.keys():
	if i not in result_fin: 
		result_fin[i]=0
	if i in result2:
		result_fin[i]+=result2[i]


print(result_fin)
lst = list(result_fin.keys())
plt.plot(lst, [result_fin[lst[i]] for i in range(5)])
plt.ylabel('tag')
plt.savefig('dic_stat.jpg')