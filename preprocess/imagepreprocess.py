import os
os.environ["CUDA_VISIBLE_DEVICES"]='1'
import  numpy as np
from torchvision.models.alexnet import alexnet
from resnet import resnet34
import scipy.io as sio
from readimage import ReadImage
import torch

dataset_name = 'mirflickr25k' 

# mirflickr25k
if dataset_name is 'mirflickr25k':
    img_dir = '/home/datasets/MIRFLICKR25K/mirflickr/'
    imgname_mat_url = '/home/datasets/MIRFLICKR25K/FAll/mirflickr25k-fall.mat'
    tag_mat_url = '/home/datasets/MIRFLICKR25K/features512/taglist_universal_sentence.mat'
    label_mat_url = '/home/datasets/MIRFLICKR25K/features512/labellist_universal_sentence.mat'
    query_num = 2000
    retrieval_num = 18015
    y_dim =1386
    class_num = 24
    seed = 6

batch_size = 20
retrieval_num = (retrieval_num // batch_size)*batch_size   #
train_num = 5000
x_dim = 1000
cuda = True

np.random.seed(seed)
random_index = np.random.permutation(range(query_num+retrieval_num))

if dataset_name is 'mirflickr25k':
    img_names = sio.loadmat(imgname_mat_url)['FAll']
else:
    img_names = sio.loadmat(imgname_mat_url)['imgs']

YL_idxs = []
for img_name in img_names:
    img_id = int(str(img_name[0]).split('[\'')[1].split('\']')[0].split('.')[0][2:])-1
    YL_idxs.append(img_id)
YAll = sio.loadmat(tag_mat_url)['tags'][YL_idxs,:]
LAll = sio.loadmat(label_mat_url)['labels'][YL_idxs,:]

T_tr = np.array(YAll[query_num:query_num + retrieval_num,:],dtype=np.float)
T_te = np.array(YAll[:query_num,:],dtype=np.float)
L_tr = np.array(LAll[query_num:query_num + retrieval_num,:],dtype=np.float)
L_te = np.array(LAll[:query_num,:],dtype=np.float)

### TO DO 2021.2.26

readImage = ReadImage(img_dir)
# img_model = vgg.vgg19(True, num_classes=x_dim)
img_model = resnet34(num_classes=x_dim, loss='fea512')
# I_tr:retrieval_num
I_tr = np.zeros((retrieval_num,512),dtype=np.float)
for i in range(retrieval_num // batch_size):
    idx = np.arange(query_num+i*batch_size,query_num+batch_size+i*batch_size)
    img_names_batchsize = img_names[idx]
    imgs = torch.zeros(batch_size,3,224,224)
    for j in range(batch_size):
        if dataset_name is 'mirflickr25k':
            img = readImage.read_img(img_names_batchsize[j][0][0].strip())
        elif dataset_name is 'nuswide':
            img = readImage.read_img(img_names_batchsize[j][0].strip()+'/'+img_names_batchsize[j][1].strip())
        elif dataset_name is 'coco2014':
            img = readImage.read_img(img_names_batchsize[j].strip())
        elif dataset_name is 'iaprtc12':
            img = readImage.read_img(img_names_batchsize[j].strip())
        imgs[j,:,:,:] = img
    if cuda:
        imgs = imgs.cuda()
        img_model = img_model.cuda()
    img_features = img_model(imgs)
    img_features = img_features.cpu().detach().numpy()
    I_tr[i*batch_size:(i*batch_size+batch_size),:] = img_features

#I_te:query_num
I_te = np.zeros((query_num,512),dtype=np.float)
for i in range(query_num // batch_size):
    idx = np.arange(i*batch_size,batch_size+i*batch_size)
    img_names_batchsize = img_names[idx]
    imgs = torch.zeros(batch_size,3,224,224)
    for j in range(batch_size):
        if dataset_name is 'mirflickr25k':
            img = readImage.read_img(img_names_batchsize[j][0][0].strip())
        elif dataset_name is 'nuswide':
            img = readImage.read_img(img_names_batchsize[j][0].strip()+'/'+img_names_batchsize[j][1].strip())
        elif dataset_name is 'coco2014':
            img = readImage.read_img(img_names_batchsize[j].strip())
        elif dataset_name is 'iaprtc12':
            img = readImage.read_img(img_names_batchsize[j].strip())
        imgs[j,:,:,:] = img
    if cuda:
        imgs = imgs.cuda()
        img_model = img_model.cuda()
    img_features = img_model(imgs)
    img_features = img_features.cpu().detach().numpy()
    I_te[idx,:] = img_features

sampleInds = np.random.permutation(range(retrieval_num))[:5000] + 1

sio.savemat('/home/datasets/MIRFLICKR25K/features512/' + dataset_name + '512.mat',mdict={'I_tr': I_tr,'I_te':I_te,'T_tr':T_tr,'T_te':T_te,'L_tr':L_tr,'L_te':L_te,'sampleInds':sampleInds})












