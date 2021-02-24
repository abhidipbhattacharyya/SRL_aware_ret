import torch
from torch.utils.data import Dataset
#import h5py
import json
import re
import os
from tsv_utils import *
import numpy as np
import nltk
#from flickr_utils import process_flickr_file

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

class CaptionDataset(Dataset):
    def __init__(self, raw_data, tsv_data, w_vocab, srl_vocab, bio_vocab, mode, opt):#(self, data_json, tsv_data, word_dic, SRL_dic,img_fname, mode = 'train', sen_index = -1, flicker_all_cap = None,transform = None, max_len = 100, srl_training= False, bio_dic = None,label_dic= None):
        self.raw_data = raw_data#json.load(open(opt.data_json, "r"))
        self.sen_index = opt.sen_index
        self.mode = mode
        self.proposition_per_sen = 3


        self.w_vocab = w_vocab
        self.srl_vocab = srl_vocab
        self.bio_vocab = bio_vocab
        #self.max_len = opt.max_len

        img_data = tsv_data
        self.imgid2img = {}
        for img_datum in img_data:
            #print(img_datum['img_id'])
            self.imgid2img[img_datum['img_id']+'.jpg'] = img_datum


        #if mode!='train':


        # Filter out the dataset
        #img_fname = '/media/abhidip/2F1499756FA9B115/data/flickr/entity/flickr30k_entities-master/'
        img_file = os.path.join(opt.data_path,self.mode+'.txt')
        with open(img_file, 'r') as l:
            lines = l.readlines()
        img_list = [l.strip()+'.jpg' for l in lines]
        data = []
        for datum in self.raw_data:
            image_name = datum['image']
            if  image_name in self.imgid2img and image_name in img_list:
                #print(image_name)

                data.append(datum)


        self.frq_prop = []
        self.data = self.process_data(data)

        print(len(self.data))
        print(len(data))
        #print(self.data[0])
        #print(self.data[1])
        #print(self.data[2])




    def __len__(self):
        return len(self.data)

    def process_single_sen(self, dsen, image):

        p_data = []
        p_tag =[]
        sen = dsen["sentence"]
        tags = dsen["tags"]
        vn_tokens = dsen["vn_tokens"]
        tokens = vn_tokens.split()#sen.split(" ")
        tokens = [tok.lower().strip() for tok in tokens]
        #tokens = nltk.word_tokenize(sen.lower())
        #if len(tokens)>= self.max_len-2 :
            #tokens = tokens[:self.max_len-2]
        tokens = ['<start>'] + tokens + ['<end>']
        enc_caption = [self.w_vocab(token) for token in tokens]

        self.frq_prop.append(len(tags))

        for i,tag_seq in enumerate(tags):
            if self.proposition_per_sen == i:
                break
            tag = tag_seq["tag"].split(" ")

            #if len(tag)>= self.max_len-2 :
                #tag = tag[:self.max_len-2]
            tag = ['<start>'] + tag + ['<end>']
            #print("===sen:{}".format(tokens))
            #print(len(enc_caption))
            #print("====tag:{}".format(tag))
            enc_tag = [self.bio_vocab(t) for t in tag]
            #print(len(enc_tag))
            assert len(enc_tag) == len(enc_caption)

            for i in range(len(tag_seq['bboxes'])):
                tag_seq['bboxes'][i]['en_srl'] = self.srl_vocab(tag_seq['bboxes'][i]['SRL'])#.get(tag_seq['bboxes'][i]['SRL'],def_srl)

            #this_proposal = {''}
            p_tag.append(enc_tag)
            p_data.append(tag_seq['bboxes'])

        return {'image':image, 'caption':sen, 'tokens': enc_caption, 'propositions':(p_data, p_tag) }

    def process_data(self, old_data):
        #sort(key=lambda x: len(x[1]), reverse = True)
        data = []
        if self.sen_index in range(5):
            print("using {}th sentence from each image".format(self.sen_index))
        else:
            print("using all sentences")

        for d in old_data:
            if self.sen_index in range(5):
                image = d["image"]
                dsen = d["sentences"][self.sen_index]
            #for dsen in d["sentences"]:
                pdata = self.process_single_sen(dsen, image)
                data.append(pdata)
            else:
                for dsen in d["sentences"]:
                    image = d["image"]

                #for dsen in d["sentences"]:
                    pdata = self.process_single_sen(dsen, image)
                    #print("=========", pdata)
                    data.append(pdata)


        return data

    def process_box(self, roi_boxes, ac_boxes):
        box_type = [self.srl_vocab('UNK')]*len(roi_boxes)

        gt_boxes = [ac["box"] for ac in ac_boxes if len(ac["box"]) >0]
        if len(gt_boxes) > 0:
            ious, listOfCordinates, neg_cord = calc_ious(roi_boxes,gt_boxes)
            for cord in listOfCordinates:
                box_type[cord[0]] = ac_boxes[cord[1]]['en_srl']

            for ncord in neg_cord:
                if box_type[ncord[0]] == self.srl_vocab('UNK'):
                    box_type[ncord[0]] = self.srl_vocab('O')

        return box_type

    def __getitem__(self, index):
        datum = self.data[index]
        #print(datum)
        #img features
        image_name = datum['image']
        img_info = self.imgid2img[image_name]
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()

        #print(datum["bboxes"])
        #print(datum["tagged_token"])
        #print(datum["tag"].split(" "))
        #print(datum["tokens"])
        box_type =  torch.zeros(self.proposition_per_sen,len(boxes)).long()
        word_type = torch.zeros(self.proposition_per_sen, len(datum['tokens'])).long()
        box_prop, word_prop = datum['propositions']
        no_of_prop = len(box_prop)

        for i,p in enumerate(box_prop):
            box_type[i,:] =  torch.LongTensor(self.process_box(boxes,p))

        for i,p in enumerate(word_prop):
            word_type[i,:] = torch.LongTensor(p)
        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        #caplen = torch.LongTensor([datum["len"]])
        box_type = torch.LongTensor(box_type)#.unsqueeze(dim=-1)
        target = torch.Tensor(datum['tokens'])
        feats = torch.FloatTensor(feats)


        #all_captions = torch.LongTensor(self.caption_map[image_name])
        return feats, torch.FloatTensor(boxes), box_type, target, index, no_of_prop, word_type


def collate_fn(data):
    # Sort input data by decreasing lengths; why? apparent below


    data.sort(key=lambda x: len(x[3]), reverse=True)
    #print("data size {}".format(len(data)))
    #print(data)
    images, boxes, box_type, captions, ids, no_of_prop, w_type = zip(*data)

    images = torch.stack(images, 0)
    boxes = torch.stack(boxes, 0)
    box_type = torch.stack(box_type,0)

    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    w_type_padded = torch.zeros(len(captions), len(w_type[0]), max(lengths)).long()

    for i,cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
        w_type_padded[i,:,:end] = w_type[i][:end]

    #print(caplen)
    return images, boxes, box_type, targets, lengths, ids, no_of_prop, w_type_padded

def get_dataloader(train_data, val_data, opt):
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, collate_fn = collate_fn, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, collate_fn=collate_fn, pin_memory=True)

    return train_loader, val_loader

def get_test_loader(test_data, opt):
    test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, collate_fn=collate_fn, pin_memory=True)

    return test_loader

if __name__ == '__main__':
    datafile = '/media/abhidip/2F1499756FA9B115/data/flickr/abhidip_splits/flickrdata_VN_SRL_martha_BBOX.json'
    tsv_path = '/media/abhidip/2F1499756FA9B115/data/flickr/image_feat/'
    flickr_file = '/media/abhidip/2F1499756FA9B115/data/flickr/flickr30k/results_20130124.token'

    img_data = []
    tsv_path1 = os.path.join(tsv_path, "train.tsv")
    tsv_path2 = os.path.join(tsv_path, "dev.tsv")
    tsv_path3 = os.path.join(tsv_path, "test.tsv")
    img_data.extend(load_obj_tsv(tsv_path1, topk=None))
    img_data.extend(load_obj_tsv(tsv_path2, topk=None))
    img_data.extend(load_obj_tsv(tsv_path3, topk=None))

    dataset = CaptionDataset(datafile, img_data, 'word_dic_A.txt', 'srl_dic_A.txt', mode = 'train',flicker_all_cap = flickr_file)
    img, boxes, box_type, caption, caplen = dataset[9751]
    print(box_type)
