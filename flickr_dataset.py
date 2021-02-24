"""Data provider"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
from tsv_utils import *
from flicker30k_entities_utils import *
from vocab import Vocabulary, from_txt_vocab
from tqdm import tqdm

enitity_root = '/media/abhidip/2F1499756FA9B115/data/flickr/entity/flickr30k_entities-master/Sentences'
bb_root = '/media/abhidip/2F1499756FA9B115/data/flickr/entity/flickr30k_entities-master/Annotations'

def calc_ious(ex_rois, gt_rois):
    gt_rois = np.asarray(gt_rois,  dtype=np.float32)
    #print(gt_rois)
    ex_area = (1. + ex_rois[:,2] - ex_rois[:,0]) * (1. + ex_rois[:,3] - ex_rois[:,1])
    gt_area = (1. + gt_rois[:,2] - gt_rois[:,0]) * (1. + gt_rois[:,3] - gt_rois[:,1])
    area_sum = ex_area.reshape((-1, 1)) + gt_area.reshape((1, -1))

    lb = np.maximum(ex_rois[:,0].reshape((-1, 1)), gt_rois[:,0].reshape((1, -1)))
    rb = np.minimum(ex_rois[:,2].reshape((-1, 1)), gt_rois[:,2].reshape((1, -1)))
    tb = np.maximum(ex_rois[:,1].reshape((-1, 1)), gt_rois[:,1].reshape((1, -1)))
    ub = np.minimum(ex_rois[:,3].reshape((-1, 1)), gt_rois[:,3].reshape((1, -1)))

    width = np.maximum(1. + rb - lb, 0.)
    height = np.maximum(1. + ub - tb, 0.)
    area_i = width * height
    area_u = area_sum - area_i
    ious = area_i / area_u
    result = np.where( ious >= 0.5)
    listOfCordinates = list(zip(result[0], result[1]))
    neg_res= np.where( ious <= 0.3)
    neg_cord = list(zip(neg_res[0], neg_res[1]))
    #print(listOfCordinates)
    return ious, listOfCordinates, neg_cord


def process_box(roi_boxes, ac_boxes, seq_len, opt = None):

    #box_type = [self.srl_dic['UNK']]*len(roi_boxes)
    def_att =  [0]*seq_len
    box_type  = [def_att]*len(roi_boxes)

    gt_boxes = [ac["box"] for ac in ac_boxes if len(ac["box"]) >0]
    if len(gt_boxes) > 0:
        ious, listOfCordinates, neg_cord = calc_ious(roi_boxes,gt_boxes)
        if opt == None or opt.attn_guide == 'normal':
            for cord in listOfCordinates:
                box_type[cord[0]] = ac_boxes[cord[1]]['attn']
        else:
            sum_of_rows = ious.sum(axis=1)
            #print(ious)
            normalized_array =ious / sum_of_rows[:, np.newaxis]
            assert len(normalized_array) == 36

            mat = []
            for j in range(len(ac_boxes)):
                mat.append([float(x) for x in ac_boxes[j]['attn']])
            mat = np.array(mat)
            #ious -> 36 x nb, mat-> nb x nw, box_type 36 x nw
            box_type = np.matmul(ious, mat)
            '''
            for i in range(len(normalized_array)):
                for j in range(len(normalized_array[i])):
                    tobeadded =  [ious[i][j]* float(x) for x in ac_boxes[j]['attn']]
                    box_type[i] = [sum(x) for x in zip(box_type[i], tobeadded)]
            '''
    return box_type



def get_bboxinfo(annotation, name):
    if name in annotation['boxes']:
        bboxes = annotation['boxes'][name]
    else:
        bboxes = []
    nobox = name in annotation['nobox']
    scene = name in annotation['scene']
    return bboxes,nobox, scene


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, tsv_data, data_split, vocab, opt = None):
        self.vocab = vocab
        self.opt = opt
        #loc = data_path + '/'

        # Image features
        img_data = tsv_data
        self.imgid2img = {}
        for img_datum in img_data:
            #print(img_datum['img_id'])
            self.imgid2img[img_datum['img_id']] = img_datum

        #self.length = len(self.captions)
        #--filter out data
        #img_fname = '/media/abhidip/2F1499756FA9B115/data/flickr/entity/flickr30k_entities-master/'
        img_file = os.path.join(data_path, data_split+'.txt')
        with open(img_file, 'r') as l:
            lines = l.readlines()
        img_list = [l.strip() for l in lines]
        data = self.process(img_list)

        self.data = []
        for d in data:
            if d['image'] in self.imgid2img:
                self.data.append(d)
        self.length = len(self.data)
        print("data lenght {}".format(self.length))


    def process(self, data):
        data_all = []
        vocab = self.vocab
        for image_name in tqdm(data):
            i=0
            enitity_sentence = os.path.join(enitity_root, image_name+'.txt')
            bb_xml = os.path.join(bb_root, image_name+ '.xml')

            sentence_data5 = get_sentence_data(enitity_sentence)
            bb_data =  get_annotations(bb_xml)

            for i in range(5):
                sentence_data = sentence_data5[i]
                tokens = sentence_data['sentence'].lower().strip().split(" ")
                tokens = ['<start>'] + tokens + ['<end>']
                enc_caption = [vocab(token) for token in tokens]

                bbox_info = []
                for p in sentence_data['phrases']:
                    name = p['phrase_id']
                    #print(  p['phrase_type'])
                    bboxes, nobox, scene = get_bboxinfo(bb_data, name)
                    #bbox_info[name] = bboxes, nobox, scene, p['first_word_index'], p['phrase_type'][0], p['phrase']
                    index_st = p['first_word_index']+1 #+1 is for start token
                    index_end = index_st+len(p['phrase'].split(" ")) -1

                    attention_guide = [0]*len(tokens)
                    for iix in range(len(p['phrase'].split(" "))):
                        attention_guide[index_st+iix] =1

                    #attn = [1]*len(p['phrase'].split(" "))
                    #attention_guide[index_st:index_end] = attn
                    #print("tpk len:{}, attention_guide:{}, attn:{}, st:{}, end{}".format( len(tokens), len(attention_guide), len(attn),index_st,index_end))
                    assert len(attention_guide) == len(tokens)
                    for t in bboxes:
                        b = {'box': t, 'id':name, 'attn':attention_guide}
                        bbox_info.append(b)

                dp = {'image':image_name, 'caption':sentence_data["sentence"], 'tokens': enc_caption, 'id':i, 'bboxes':bbox_info}
                data_all.append(dp)

        return data_all


    def __getitem__(self, index):
        # handle the image redundancy
        datum = self.data[index]
        #print(datum)
        #img features
        image_name = datum['image']
        img_info = self.imgid2img[image_name]
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()

        attention_guide = process_box(boxes,datum["bboxes"],len(datum['tokens']),opt = self.opt)# 36 x seq_len

        # Convert caption (string) to word ids.
        target = torch.Tensor(datum['tokens'])
        feats = torch.FloatTensor(feats)
        attention_guide = torch.FloatTensor(attention_guide)

        #print(attention_guide.size())
        #print(feats.size())
        #print(target.size())
        #print("=========================")

        return feats, target, index, attention_guide

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, attn = zip(*data)

    #num_obj = attn.size(1)
    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    attention_guide = torch.zeros(len(captions), 36, max(lengths)).float()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
        attention_guide[i,:,:end ] = attn[i][:,:end]

    return images, targets, lengths, ids, attention_guide


def get_precomp_loader(data_path, tsv_data, data_split, vocab, opt=None, batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = PrecompDataset(data_path, tsv_data, data_split, vocab, opt = opt)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              collate_fn=collate_fn)
    return data_loader


def get_loaders(datafolder, tsv_data, vocab, batch_size=100, workers=2, opt=None):
    #dpath = os.path.join(opt.data_path, data_name)
    train_loader = get_precomp_loader(datafolder, tsv_data, 'train', vocab, opt,
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(datafolder, tsv_data, 'dev', vocab, opt,
                                    batch_size, False, workers)
    return train_loader, val_loader


def get_test_loader(split_name, datafolder, tsv_data, vocab, batch_size=100, workers=2, opt=None):

    test_loader = get_precomp_loader(datafolder, tsv_data, split_name, vocab, opt,
                                     batch_size, False, workers)
    return test_loader

if __name__ == '__main__':
    datafolder = '/media/abhidip/2F1499756FA9B115/data/flickr/entity/flickr30k_entities-master/'
    tsv_path = '/media/abhidip/2F1499756FA9B115/data/flickr/image_feat/'


    img_data = []
    tsv_path1 = os.path.join(tsv_path, "train.tsv")
    tsv_path2 = os.path.join(tsv_path, "dev.tsv")
    tsv_path3 = os.path.join(tsv_path, "test.tsv")
    img_data.extend(load_obj_tsv(tsv_path1, topk=None))
    img_data.extend(load_obj_tsv(tsv_path2, topk=None))
    img_data.extend(load_obj_tsv(tsv_path3, topk=None))
    vocab = from_txt_vocab("word_dic_A.txt")
    test_loader =  get_test_loader("dev", datafolder, img_data, vocab, batch_size= 32)
    for  (images, targets, lengths, ids, attention_guide)  in test_loader:
        print(images.size())
        print(targets.size())
        #print(lengths.size())
        print(attention_guide.size())
        break
