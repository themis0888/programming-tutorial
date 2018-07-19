"""
This program distribute every images in each categories to train data and validation data.

you have 6 category of images in below directory
/base_dir/source_dir
 ├── AD
 ├── ADULT
 ├── HATRED
 ├── ILLEGALITY
 ├── NORMAL
 └── SEMI_ADULT

after you run 'python file_devider.py', you will get 
/base_dir/data_dir
├── train
│   ├── AD
│   ├── ADULT
│   ├── HATRED
│   ├── ILLEGALITY
│   ├── NORMAL
│   └── SEMI_ADULT
└── val
    ├── AD
    ├── ADULT
    ├── HATRED
    ├── ILLEGALITY  
    ├── NORMAL
    └── SEMI_ADULT


And I'm pretty sure that there must be much more faster method in pytorch 
which does exactly same thing. :D 
"""

from os import walk
from PIL import Image
import os, shutil, random, sys, struct

def img_convert(fname, src_dir, dest_dir):
    size = 256, 256
    im = Image.open(src_dir + fname)
    im.load()
    # This code convert readible image to jpeg format. 
    # Which means you don't need to check the file type.  
    im.thumbnail(size, Image.ANTIALIAS)        
    im.convert('RGB').save(dest_dir + fname, 'JPEG')
    return im.size, im.format

base_dir_o = '~/../shared/danbooru2017/512px/'
dest_dir_o = '~/../shared/danbuuru2017/256px/'

# I'm sure this way is pretty stupid, but after you do this once, it would run fast 
# because system store this list on the cache.
# randomly shffle the files and divide 

for i in range(500):
    base_dir = base_dir_o + '{0:04d}'.format(i)
    dest_dir = dest_dir_o + '{0:04d}'.format(i)

    if (not os.path.exists(dest_dir)):
        os.mkdir(dest_dir)

    print('From {}'.format(base_dir))
    print('To {}'.format(dest_dir))

    dir_lst = os.listdir(base_dir)
    random.shuffle(dir_lst)
    #num_file = 1000
    num_file = len(dir_lst)
    idx = 0

    for name in dir_lst:
        try:
            image_size, img_format = img_convert(name, base_dir, dest_dir)
        except IOError:
            idx += 1
            continue
        except ValueError:
            idx += 1
            continue
        else: 
            if idx%100 == 0:
                print('{:.4f}% \tDone'.format(idx/num_file))
            idx += 1
        #if idx > 1000: break
