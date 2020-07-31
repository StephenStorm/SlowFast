import os
import re


valfile = './data_process/data/vallist.txt'

test_csv = './data_process/data/test.csv'

kinetics = '/home/yhzn/dataset/Kinetics'
# kinetics = '/home/stephen/tmp'

with open(valfile, 'r') as vallist, open(test_csv, 'w') as test:
    for line in vallist:
        path_label = line.split(' ')
        path = path_label[0]
        label = path_label[1]
        paths = path.split('/')
        video_dir = paths[-2].replace('_', ' ')
        video_name = paths[-1].split('.')[0]
        # print('{0}\{1}'.format(video_dir, video_name))
        target_dir = os.path.join(kinetics, video_dir)
        try:
            video_names = os.listdir(target_dir)
        except FileNotFoundError:
            continue
        # print(video_names)
        for name in video_names:
            if re.match(video_name + '*', name):
                video_path = os.path.join(target_dir, name)
                test.write('{},{}'.format(video_path, label))
                