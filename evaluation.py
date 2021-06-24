# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Evaluation"""

from __future__ import print_function
import os
import json
import sys
from flickr_dataset import get_test_loader
import time
import numpy as np
from vocab import Vocabulary, deserialize_vocab, from_txt_vocab
import torch
from GCN_model_ret import SCAN, xattn_score_t2i, xattn_score_i2t, xattn_score
#from model_bfan import SCAN, xattn_score_t2i, xattn_score_i2t, xattn_score
#from model import SCAN, xattn_score_t2i, xattn_score_i2t
from collections import OrderedDict
import time
from flickr_dataset_All import *
from torch.autograd import Variable

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None
    cap_lens = None

    max_n_word = 0
    for i, (images, boxes, box_type, targets, lengths, ids, no_of_prop, w_type_padded) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    for i, (images, boxes, box_type, captions, lengths,ids, no_of_prop, w_type_padded) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger
        bsize = images.size(0)
        no_of_prop =  w_type_padded.size(1)
        # compute the embeddings
        img_emb, cap_emb, cap_len = model.forward_emb(images, boxes, box_type, captions, lengths,no_of_prop, w_type_padded, volatile=True)
        img_emb1 = img_emb.view(bsize, no_of_prop, img_emb.size(1), img_emb.size(2) )
        cap_emb1 = cap_emb.view(bsize, no_of_prop, cap_emb.size(1), cap_emb.size(2) )
        #print(img_emb)
        if img_embs is None:
            if img_emb1.dim() > 3:
                img_embs = np.zeros((len(data_loader.dataset),no_of_prop, img_emb.size(1), img_emb.size(2)))
            else:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), no_of_prop, max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids] = img_emb1.data.cpu().numpy().copy()
        cap_embs[ids,:,:max(lengths),:] = cap_emb1.data.cpu().numpy().copy()

        for j, nid in enumerate(ids):
            cap_lens[nid] = lengths[j]

        # measure accuracy and record loss
        model.forward_loss(img_emb, cap_emb, cap_len)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
            #print('Test: [{0}/{1}]\t'
            #        '{e_log}\t'
            #        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #        .format(
            #            i, len(data_loader), batch_time=batch_time,
            #            e_log=str(model.logger)))
        del images, captions
    return img_embs, cap_embs, cap_lens


def evalrank(model_path, tsv_data, data_json, data_path=None, split='test', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    print("loading model from {}".format(model_path))
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    print("model was saved at {}".format(checkpoint['epoch']))
    print(opt)
    if data_path is not None:
        opt.data_path = data_path

    # load vocabulary used by the model

    w_vocab = from_txt_vocab("word_dic_A.txt")
    opt.w_vocab_size = len(w_vocab)

    srl_vocab = from_txt_vocab("srl_dic_A.txt")
    opt.srl_vocab_size = len(srl_vocab)

    bio_vocab = from_txt_vocab("bio_srl_A.txt")
    opt.bio_vocab_size = len(bio_vocab)

    # construct model
    model = SCAN(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    raw_data = json.load(open(data_json, "r"))
    test_dataset = CaptionDataset(raw_data, tsv_data , w_vocab, srl_vocab,  bio_vocab, 'test', opt)
    data_loader = get_test_loader(test_dataset, opt)

    print('Computing results...')
    img_embs, cap_embs, cap_lens = encode_data(model, data_loader)
    #print('Images: %d, Captions: %d' %(img_embs.shape[0] / 5, cap_embs.shape[0]))


    if not fold5:
        # no cross-validation, full evaluation
        #img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
        start = time.time()
        if opt.cross_attn == 't2i':
            sims = shard_xattn_t2i(img_embs, cap_embs, cap_lens, opt, shard_size=128)
        elif opt.cross_attn == 'i2t':
            sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, opt, shard_size=128)
        elif opt.cross_attn == 'both':
            sims = shard_xattn(img_embs, cap_embs, cap_lens, opt, shard_size=128)
        else:
            raise NotImplementedError
        end = time.time()
        print("calculate similarity time:", end-start)

        r, rt = i2t_any(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
        ri, rti = t2i_any(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text:%.1f %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image:%.1f %.1f %.1f %.1f %.1f %.1f"% ri)


        r, rt = i2t_exact(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
        ri, rti = t2i_exact(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        print("rsum: %.1f" % rsum)
        print("Average i2t Recall: %.1f" % ar)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f %.1f" % r)
        print("Average t2i Recall: %.1f" % ari)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)

        filewrite(img_embs, cap_embs, cap_lens, sims, test_dataset.data)
    else:
        # 5fold cross-validation, only for MSCOCO
        results = []
        for i in range(5):
            img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
            cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
            cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]
            start = time.time()
            if opt.cross_attn == 't2i':
                sims = shard_xattn_t2i(img_embs_shard, cap_embs_shard, cap_lens_shard, opt, shard_size=128)
            elif opt.cross_attn == 'i2t':
                sims = shard_xattn_i2t(img_embs_shard, cap_embs_shard, cap_lens_shard, opt, shard_size=128)
            else:
                raise NotImplementedError
            end = time.time()
            print("calculate similarity time:", end-start)

            r, rt0 = i2t(img_embs_shard, cap_embs_shard, cap_lens_shard, sims, return_ranks=True)
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(img_embs_shard, cap_embs_shard, cap_lens_shard, sims, return_ranks=True)
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        print("Average i2t Recall: %.1f" % mean_metrics[11])
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[:5])
        print("Average t2i Recall: %.1f" % mean_metrics[12])
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
              mean_metrics[5:10])

    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


def softmax(X, axis):
    """
    Compute the softmax of each element along an axis of X.
    """
    y = np.atleast_2d(X)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    return p

def process_sim(sim, no_prop):
    num_of_image = len(sim)//no_prop
    num_of_caption = len(sim[0])//no_prop
    assert len(sim)%no_prop ==0
    assert len(sim[0])%no_prop ==0
    ac_sim = np.zeros((num_of_image, num_of_caption))

    for i in range(num_of_image):
        st_i_index = i*no_prop
        for j in range(num_of_caption):
            st_j_index = j*no_prop
            sum=0
            for k in range(no_prop):
                sum+=sim[st_i_index+k][st_j_index+k]

            ac_sim[i][j] = sum
    return ac_sim

def shard_xattn(images, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = (len(images)-1)//shard_size + 1
    n_cap_shard = (len(captions)-1)//shard_size + 1

    print("{}/{}".format(n_im_shard,n_im_shard))
    no_prop = 3
    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            no_prop = im.size(1)
            im= im.view(im.size(0)*im.size(1), im.size(2), im.size(3))
            s= s.view(s.size(0)*s.size(1), s.size(2), s.size(3))
            lens = caplens[cap_start:cap_end]
            l = []
            for ll in lens:
                for _ in range(no_prop):
                    l.append(ll)

            sim,_, _ = xattn_score(im, s, l, opt, mode='dev')
            ac_sim = process_sim(sim.cpu().numpy(), no_prop)
            d[im_start:im_end, cap_start:cap_end] = ac_sim
            #break
        #break
    sys.stdout.write('\n')
    return d

def shard_xattn_t2i(images, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = (len(images)-1)//shard_size + 1
    n_cap_shard = (len(captions)-1)//shard_size + 1
    print(n_im_shard)
    print(n_cap_shard)

    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            l = caplens[cap_start:cap_end]
            sim, att = xattn_score_t2i(im, s, l, opt, mode='dev')
            d[im_start:im_end, cap_start:cap_end] = sim.cpu().numpy()
    sys.stdout.write('\n')
    return d


def shard_xattn_i2t(images, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = (len(images)-1)//shard_size + 1
    n_cap_shard = (len(captions)-1)//shard_size + 1

    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            l = caplens[cap_start:cap_end]
            sim, att = xattn_score_i2t(im, s, l, opt, mode = 'dev')
            d[im_start:im_end, cap_start:cap_end] = sim.cpu().numpy()
    sys.stdout.write('\n')
    return d


def i2t_any(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap

    images: (5N, p, n_region, d)
    Captions: (5N, p, max_n_word, d)
    caplens:5N
    sims: (5N,5N)
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        cap_st_indx = index//5
        for i in range(5 * cap_st_indx, 5 * cap_st_indx + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r3 = 100.0 * len(np.where(ranks < 3)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    print('i2t r3:{}'.format(r3))
    if return_ranks:
        return (r1, r3,  r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r3, r5, r10, medr, meanr)


def t2i_any(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap

    images: (5N, p, n_region, d)
    Captions: (5N, p, max_n_word, d)
    caplens:5N
    sims: (5N,5N)
    """
    npts = captions.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    sims = sims.T

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        img_st_indx = index//5
        for i in range(5 * img_st_indx, 5 * img_st_indx + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r3 = 100.0 * len(np.where(ranks < 3)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    print('t2i  r3:{}'.format(r3))
    if return_ranks:
        return (r1, r3, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r3, r5, r10, medr, meanr)

def i2t_exact(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation) exact match

    images: (5N, p, n_region, d)
    Captions: (5N, p, max_n_word, d)
    caplens:5N
    sims: (5N,5N)
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        #for i in range(5 * index, 5 * index + 5, 1):
        tmp = np.where(inds == index)[0][0]
        if tmp < rank:
            rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


    sims = sims.T

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        #for i in range(5 * index, 5 * index + 5, 1):
        tmp = np.where(inds == index)[0][0]
        if tmp < rank:
            rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]


    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i_exact(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search) exact match
    images: (5N, p, n_region, d)
    Captions: (5N, p, max_n_word, d)
    caplens:5N
    sims: (5N,5N)
    """
    npts = captions.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        #for i in range(5 * index, 5 * index + 5, 1):
        tmp = np.where(inds == index)[0][0]
        if tmp < rank:
            rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]


    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def filewrite(images, captions, caplens, sims, rawdata ):
    """
    Images->Text (Image Annotation) any of the 5 caption


    images: (5N, p, n_region, d)
    Captions: (5N, p, max_n_word, d)
    caplens:5N
    sims: (5N,5N)
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)

    image_caption = []
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        image_caption.append(inds[:5]) #take  top 5


    ret_result = {}
    for i, ind in enumerate(image_caption):
        data =  rawdata[i]
        image_name  = data["image"]
        caption = data["caption"]
        ret_cap = [rawdata[j]["caption"] for j in ind]
        img_id = image_name + "_" + str(i%5)+".jpg"
        ret_result[img_id] = {"image":img_id, "caption":caption, "ret_cap":ret_cap }


    npts = captions.shape[0]
    ranks = np.zeros(npts)
    sims = sims.T
    caption_image = []
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        caption_image.append(inds[:5])

    for i, ind in enumerate(caption_image):
        data =  rawdata[i]
        image_name  = data["image"]
        caption = data["caption"]
        ret_img = [rawdata[j]["image"]+"_"+str(j%5)+".jpg" for j in ind]
        img_id = image_name + "_" + str(i%5)+".jpg"
        ret_result[img_id]["ret_img"] = ret_img


    with open("ret_op.json", "w") as f:
        json.dump(list(ret_result.values()), f, indent=4)
