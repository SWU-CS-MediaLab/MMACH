'''
author:xiaotao
time:2020.11.10-
about: transform the original text of datasets (i.e. mirflckr25k, nuswide, microsoft coco2014 and iaprtc12) to 512-dimensional features  with pretrained transformers.
'''

import os
from transformers import AutoTokenizer, AutoModel
import numpy as np
import scipy.io as sio

tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-512_A-8")  # google/bert_uncased_L-4_H-512_A-8
model = AutoModel.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
def myTransformer(mytext):
    inputs = tokenizer(mytext, return_tensors="pt")
    _, output = model(**inputs)
    return output

def gen_mirflickr25k_512tag(dataset_name):
    if dataset_name.lower() == 'mirflickr25k':
        path='/home/datasets/MIRFLICKR25K/labels25000/my_labels/'
        files=os.listdir(path)
        text_num=len(files)
        texts_features=np.zeros((text_num,512),dtype=np.float)
        count=0



        for file in files:
            print(file)
            text_id=file.split('.')[0][5:]

            position=path+file
            text=''
            with open(position,'r') as f:
                for line in f.readlines():
                    text=text+line.strip('\n')+' '

            feature512=myTransformer(text)
            texts_features[int(text_id)-1,:]=feature512.detach().numpy()
            
            print(count)
            count = count + 1
        sio.savemat('/home/datasets/MIRFLICKR25K/features512/labellist.mat', mdict={'labels': texts_features})

    else:
        raise ValueError("there is no dataset name is %s" % dataset_name)


if __name__ == '__main__':
    gen_mirflickr25k_512tag('mirflickr25k')


