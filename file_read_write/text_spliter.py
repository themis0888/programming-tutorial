import os

write_path = 'txt'
if not os.path.exists(write_path): os.mkdir(write_path)

read_file = open('../../file_name.txt', 'r', encoding='utf-8')

lines = read_file.readlines()

for idx, line in enumerate(lines):
    write_file_name ='{0:02d}.txt'.format(idx+1)
    if idx < 10: print(write_file_name)
    write_file = open(os.path.join(write_path, write_file_name), 'w', encoding='utf-8')
    write_file.write(line)
    write_file.close()

