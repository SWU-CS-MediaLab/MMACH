# Multi-label modality enhanced attention based self-supervised deep cross-modal hashing (MMACH)

Pytorch implementation of paper 'Multi-label modality enhanced attention based self-supervised deep
cross-modal hashing'.

## Abstract

The recent deep cross-modal hashing (DCMH) has achieved superior performance in effective and
efficient cross-modal retrieval and thus has drawn increasing attention. Nevertheless, there are still
two limitations for most existing DCMH methods: (1) single labels are usually leveraged to measure the
semantic similarity of cross-modal pairwise instances while neglecting that many cross-modal datasets
contain abundant semantic information among multi-labels. (2) several DCMH methods utilized the
multi-labels to supervise the learning of hash functions. Nevertheless, the feature space of multilabels
suffers the weakness of sparse, resulting in sub-optimization for the hash functions learning.
Thus, this paper proposed a multi-label modality enhanced attention-based self-supervised deep
cross-modal hashing (MMACH) framework. Specifically, a multi-label modality enhanced attention
module is designed to integrate the significant features from cross-modal data into multi-labels feature
representations, aiming to improve its completion. Moreover, a multi-label cross-modal triplet loss
is defined based on the criterion that the feature representations of cross-modal pairwise instances
with more common categories should preserve higher semantic similarity than other instances. To
the best of our knowledge, the multi-label cross-modal triplet loss is the first time designed for
cross-modal retrieval. Extensive experiments on four multi-label cross-modal datasets demonstrate the
effectiveness and efficiency of our proposed MMACH. Moreover, the MMACH also achieved superior
performance and outperformed several state-of-the-art methods on the task of cross-modal retrieval.

------

Please cite our paper if you use this code in your own work:

@article{zou2022multi,
  title={Multi-label modality enhanced attention based self-supervised deep cross-modal hashing},
  author={Zou, Xitao and Wu, Song and Zhang, Nian and Bakker, Erwin M},
  journal={Knowledge-Based Systems},
  volume={239},
  pages={107927},
  year={2022},
  publisher={Elsevier}
}

---
### Dependencies 
you need to install these package to run
- visdom 0.1.8+
- pytorch 1.0.0+
- tqdm 4.0+  
- python 3.5+
----

### Dataset

we implement our method on dataset Mirflckr25K:

(1) please download the original image-text data of Mirflckr25K from http://press.liacs.nl/mirflickr/mirdownload.html  and put it under the folder /dataset/data/.

(2) please download the mirflickr25k-fall.mat, mirflickr25k-iall.mat, mirflickr25k-lall.mat and mirflickr25k-yall.mat from https://pan.baidu.com/s/1FX82NhdtnTeARcgmqxYCag 
(提取码：imk4) and put them under the folder /dataset/data/.

### How to run
 
 The implementation of MMACH is similar to the implementation of MESDCH (https://github.com/SWU-CS-MediaLab/MESDCH).

If you have any problems, please feel free to contact Xitao Zou (xitaozou@mail.swu.edu.cn).
