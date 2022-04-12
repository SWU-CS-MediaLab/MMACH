import os
import scipy.io as sio

original_labels_path = '/home/datasets/MIRFLICKR25K/labels25000/mirflickr25k_annotations_v080/'
target_labels_path = '/home/datasets/MIRFLICKR25K/labels25000/my_labels/'
labels_path = '/home/datasets/MIRFLICKR25K/labels25000/labels24.txt'
# text_path='/home/datasets/MIRFLICKR25K/mirflickr/meta/tags/'
# files=os.listdir(text_path)
# for file in files:
#     text_id = file.split('.')[0][4:]
#     open(target_labels_path+'label'+text_id+'.txt', 'w')
#
classes = []
with open(labels_path, 'r') as f_l:
    for label in f_l.readlines():
        classes.append(label.strip('\n'))

original_labels = os.listdir(original_labels_path)

for label in classes:
    for original_l in original_labels:
        if original_l.find('_') > 0:
            filename = original_l
            labelname = original_l.split('_')[0]
        else:
            filename = original_l
            labelname = original_l.split('.')[0]

        myids = []
        with open(original_labels_path + filename, 'r') as o_l:
            for id in o_l.readlines():
                myids.append(id.strip('\n'))

        for myid in myids:
            myfile_path = target_labels_path + 'label' + myid + '.txt'
            flag = 0
            with open(myfile_path, 'r') as m_o:
                for ml in m_o.readlines():
                    if ml.strip('\n')==labelname:
                        flag = 1
                        break
            if flag == 0:
                with open(myfile_path, 'a') as m_p:
                    m_p.write(labelname + '\n')






