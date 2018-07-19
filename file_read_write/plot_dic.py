import collections, itertools
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os 
import operator


temp_100 = {'1girl': 345674, 'bangs': 37471, 'greyscale': 39098, 'looking_at_viewer': 102725, 'monochrome': 50695, 'short_hair': 137511, 'solo': 287883, 'bad_id': 72023, 'bad_pixiv_id': 68063, 'highres': 169068, 'long_hair': 220143, 'purple_eyes': 37339, 'purple_hair': 33781, 'school_uniform': 55722, 'original': 56444, 'thighhighs': 77762, 'blush': 142716, 'comic': 42099, 'touhou': 97488, 'translated': 47231, 'panties': 44679, 'underwear': 51699, 'blue_hair': 47154, 'brown_eyes': 56534, 'serafuku': 24674, 'skirt': 86053, 'smile': 133734, 'blonde_hair': 97508, 'blue_eyes': 97282, 'hair_ornament': 60176, 'hat': 81433, 'twintails': 55694, 'wings': 27897, '1boy': 61644, 'boots': 26836, 'bow': 57729, 'breasts': 145838, 'gloves': 61352, 'heart': 23012, 'holding': 23556, 'large_breasts': 72305, 'nipples': 38840, 'nude': 24963, 'animal_ears': 44793, 'brown_hair': 90703, 'commentary_request': 25552, 'japanese_clothes': 24893, 'green_eyes': 48079, 'one_eye_closed': 24056, 'tail': 30567, 'absurdres': 33377, 'ahoge': 25505, 'eyebrows_visible_through_hair': 20407, 'long_sleeves': 38159, 'shirt': 40937, 'male_focus': 25817, 'red_eyes': 73911, 'bare_shoulders': 34732, 'jewelry': 37762, 'simple_background': 53045, 'white_background': 41454, 'barefoot': 20899, 'flower': 29384, 'food': 23303, 'green_hair': 27073, 'lying': 21074, 'multiple_girls': 112312, 'sitting': 43857, 'black_hair': 75189, 'closed_eyes': 38280, 'glasses': 25661, 'red_hair': 27388, 'translation_request': 28361, 'weapon': 38016, 'full_body': 24069, 'multiple_boys': 24319, 'standing': 24347, 'hair_ribbon': 36011, 'open_mouth': 108035, 'ribbon': 64490, 'yellow_eyes': 31647, 'hairband': 27527, 'kantai_collection': 43525, 'cleavage': 42535, 'collarbone': 20178, '2girls': 71117, 'black_legwear': 32126, 'ponytail': 33924, 'very_long_hair': 36799, 'detached_sleeves': 23990, 'necktie': 21405, 'hair_bow': 30569, 'swimsuit': 29256, 'dress': 59495, 'd': 23827, 'navel': 46941, 'braid': 27955, 'ass': 28515, 'pink_hair': 32509, 'pantyhose': 26738, 'medium_breasts': 44779, 'silver_hair': 29798}
temp_over_100k = {'1girl': 2060363, 'bangs': 228526, 'greyscale': 224236, 'looking_at_viewer': 631508, 'monochrome': 291271, 'short_hair': 813241, 'solo': 1710762, 'bad_id': 419738, 'bad_pixiv_id': 396009, 'highres': 1018512, 'long_hair': 1318516, 'purple_eyes': 224448, 'purple_hair': 201828, 'school_uniform': 327240, 'original': 336551, 'thighhighs': 466367, 'blush': 870086, 'comic': 243857, 'touhou': 573047, 'translated': 275242, 'panties': 277119, 'underwear': 320543, 'blue_hair': 276610, 'brown_eyes': 336590, 'serafuku': 144665, 'skirt': 508437, 'smile': 800738, 'blonde_hair': 580097, 'blue_eyes': 585282, 'hair_ornament': 361132, 'hat': 478805, 'twintails': 329957, 'wings': 164857, '1boy': 360072, 'boots': 158094, 'bow': 344516, 'breasts': 900500, 'gloves': 365281, 'heart': 139669, 'holding': 139520, 'large_breasts': 447962, 'nipples': 247723, 'nude': 157512, 'animal_ears': 267820, 'brown_hair': 541144, 'commentary_request': 157267, 'hair_between_eyes': 102962, 'japanese_clothes': 145433, 'upper_body': 107324, 'green_eyes': 284901, 'one_eye_closed': 142753, 'tail': 183001, 'absurdres': 203480, 'ahoge': 151627, 'eyebrows_visible_through_hair': 128146, 'jacket': 112358, 'long_sleeves': 226893, 'shirt': 245483, 'male_focus': 146755, 'red_eyes': 440672, 'bare_shoulders': 212757, 'censored': 124788, 'jewelry': 225500, 'penis': 109715, 'simple_background': 319348, 'white_background': 249868, '3girls': 109463, 'barefoot': 129134, 'flower': 172918, 'food': 136884, 'green_hair': 157873, 'lying': 132207, 'multiple_girls': 662570, 'sitting': 267050, 'black_hair': 450133, 'closed_eyes': 223995, 'glasses': 150525, 'red_hair': 161975, 'sword': 110739, 'tears': 119437, 'translation_request': 168122, 'weapon': 221675, 'full_body': 143927, 'multiple_boys': 140937, 'shoes': 114441, 'short_sleeves': 106201, 'standing': 146313, 'hair_ribbon': 215494, 'open_mouth': 651028, 'ribbon': 384798, 'yellow_eyes': 189887, 'hairband': 164088, 'kantai_collection': 263009, 'cleavage': 260152, 'collarbone': 126856, 'midriff': 104302, 'white_hair': 111443, '2girls': 422057, 'black_legwear': 196926, 'ponytail': 202858, 'very_long_hair': 220922, 'sky': 103574, 'detached_sleeves': 140977, 'necktie': 125349, 'hair_bow': 182346, 'striped': 105154, 'bikini': 122932, 'swimsuit': 180830, 'dress': 351070, 'd': 143757, 'sweat': 123441, 'elbow_gloves': 107276, 'navel': 295551, 'braid': 165659, 'ass': 182623, 'pussy': 112565, 'white_legwear': 104953, 'small_breasts': 118246, 'pink_hair': 194393, 'pantyhose': 160051, 'medium_breasts': 273282, 'silver_hair': 179452, 'pleated_skirt': 115449}

result = temp_over_100k
# result_fin = collections.OrderedDict(sorted(result_fin.items()))
result_fin = sorted(result.items(), key=operator.itemgetter(1), reverse=True)
result_fin = [(elem1, np.log(elem2)) for elem1, elem2 in result_fin]

result_fin = result_fin[:50]
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

