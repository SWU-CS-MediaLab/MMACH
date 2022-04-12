import os
from transformers import AutoTokenizer, AutoModel
import numpy as np
import scipy.io as sio

def myTransformer(mytext):
    tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-4_H-512_A-8")  # google/bert_uncased_L-4_H-512_A-8
    model = AutoModel.from_pretrained("google/bert_uncased_L-4_H-512_A-8")
    inputs = tokenizer(mytext, return_tensors="pt")
    _, output = model(**inputs)
    return output

path='/home/datasets/MIRFLICKR25K/mirflickr/meta/tags/tags21751.txt'
text=''
with open(path, 'r') as f:
    for line in f.readlines():
        text = text + line.strip('\n') + ' '
feature512=myTransformer(text)
print(feature512.size())
