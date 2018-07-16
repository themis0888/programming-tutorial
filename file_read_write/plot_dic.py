import collections, itertools
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os 
import operator


temp = {'1girl': 345674, 'bangs': 37471, 'greyscale': 39098, 'looking_at_viewer': 102725, 'monochrome': 50695, 'short_hair': 137511, 'solo': 287883, 'bad_id': 72023, 'bad_pixiv_id': 68063, 'highres': 169068, 'long_hair': 220143, 'purple_eyes': 37339, 'purple_hair': 33781, 'school_uniform': 55722, 'original': 56444, 'thighhighs': 77762, 'blush': 142716, 'comic': 42099, 'touhou': 97488, 'translated': 47231, 'panties': 44679, 'underwear': 51699, 'blue_hair': 47154, 'brown_eyes': 56534, 'serafuku': 24674, 'skirt': 86053, 'smile': 133734, 'blonde_hair': 97508, 'blue_eyes': 97282, 'hair_ornament': 60176, 'hat': 81433, 'twintails': 55694, 'wings': 27897, '1boy': 61644, 'boots': 26836, 'bow': 57729, 'breasts': 145838, 'gloves': 61352, 'heart': 23012, 'holding': 23556, 'large_breasts': 72305, 'nipples': 38840, 'nude': 24963, 'animal_ears': 44793, 'brown_hair': 90703, 'commentary_request': 25552, 'japanese_clothes': 24893, 'green_eyes': 48079, 'one_eye_closed': 24056, 'tail': 30567, 'absurdres': 33377, 'ahoge': 25505, 'eyebrows_visible_through_hair': 20407, 'long_sleeves': 38159, 'shirt': 40937, 'male_focus': 25817, 'red_eyes': 73911, 'bare_shoulders': 34732, 'jewelry': 37762, 'simple_background': 53045, 'white_background': 41454, 'barefoot': 20899, 'flower': 29384, 'food': 23303, 'green_hair': 27073, 'lying': 21074, 'multiple_girls': 112312, 'sitting': 43857, 'black_hair': 75189, 'closed_eyes': 38280, 'glasses': 25661, 'red_hair': 27388, 'translation_request': 28361, 'weapon': 38016, 'full_body': 24069, 'multiple_boys': 24319, 'standing': 24347, 'hair_ribbon': 36011, 'open_mouth': 108035, 'ribbon': 64490, 'yellow_eyes': 31647, 'hairband': 27527, 'kantai_collection': 43525, 'cleavage': 42535, 'collarbone': 20178, '2girls': 71117, 'black_legwear': 32126, 'ponytail': 33924, 'very_long_hair': 36799, 'detached_sleeves': 23990, 'necktie': 21405, 'hair_bow': 30569, 'swimsuit': 29256, 'dress': 59495, 'd': 23827, 'navel': 46941, 'braid': 27955, 'ass': 28515, 'pink_hair': 32509, 'pantyhose': 26738, 'medium_breasts': 44779, 'silver_hair': 29798}

result = temp
# result_fin = collections.OrderedDict(sorted(result_fin.items()))
result_fin = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
result_fin = [(elem1, np.log(elem2)) for elem1, elem2 in result_fin]
zip(*result_fin)
font = {'size':5}
matplotlib.rc('font', **font)
"""
fig, axs = plt.subplots(1,1)

axs.scatter(*zip(*result_fin), '.')
axs.set_yscale('log')
axs.set_ylim([0, np.maximum(np.log(result_fin))])
"""
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

