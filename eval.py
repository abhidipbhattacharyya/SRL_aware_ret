from vocab import Vocabulary, deserialize_vocab, from_txt_vocab
import evaluation
from tsv_utils import *
import os

tsv_path = '/data/abhidip/MM_SRL_data/flickr/image_feat'

img_data = []
tsv_path1 = os.path.join(tsv_path, "train.tsv")
tsv_path2 = os.path.join(tsv_path, "dev.tsv")
tsv_path3 = os.path.join(tsv_path, "test.tsv")
img_data.extend(load_obj_tsv(tsv_path1, topk=None))
img_data.extend(load_obj_tsv(tsv_path2, topk=None))
img_data.extend(load_obj_tsv(tsv_path3, topk=None))

data_path = '/data/abhidip/MM_SRL_data/flickr/entity/flickr30k_entities-master/'
model_path = '/data/abhidip/MM_SRL_data/flickr/models/SRL_aware_ret/checkpoint/gcn/model_best.pth.tar'
data_json ='/data/abhidip/MM_SRL_data/flickr/flickrdata_VN_SRL_martha_BBOX2.json'# data_json = '/media/abhidip/2F1499756FA9B115/data/flickr/abhidip_splits/flickrdata_VN_SRL_martha_BBOX2.json'

evaluation.evalrank(model_path,img_data, data_json, data_path=data_path, split="test")
