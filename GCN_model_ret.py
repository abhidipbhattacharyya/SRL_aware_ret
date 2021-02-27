# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""SCAN model"""

import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from jsd import *
from GCN_lib.Rs_GCN import Rs_GCN

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def EncoderImage(data_name, img_dim, srl_vocab_size, embed_size, precomp_enc_type='basic',
                 no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        print('creating basic image encoder')
        img_enc = EncoderImagePrecomp(
            img_dim, srl_vocab_size, embed_size, no_imgnorm)
    elif precomp_enc_type == 'GCN':
        print('creating GCN image encoder')
        img_enc = GCN_encoder(
            img_dim, srl_vocab_size, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc

class GCN_encoder(nn.Module):

    def __init__(self, img_dim,  srl_vocab_size, embed_size, use_abs=False, no_imgnorm=False, hidden_dropout_prob=0.1):
        super(GCN_encoder, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        #self.data_name = data_name

        self.fc = nn.Linear(img_dim, embed_size//2)
        self.embedding = nn.Embedding(srl_vocab_size, embed_size//2)
        self.fc2 = nn.Linear(embed_size, embed_size)

        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.init_weights()


        # GSR
        self.img_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)

        # GCN reasoning
        self.Rs_GCN_1 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_2 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_3 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.Rs_GCN_4 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)

        #if self.data_name == 'f30k_precomp':
        self.bn = nn.BatchNorm1d(embed_size)

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

        r2 = np.sqrt(6.) / np.sqrt(self.fc2.in_features +
                                  self.fc2.out_features)
        self.fc2.weight.data.uniform_(-r2, r2)
        self.fc2.bias.data.fill_(0)


    #def forward(self, images, srl):
    def forward(self, images, boxes, box_type, no_of_prop):
        """Extract image feature vectors."""

        features = self.fc(images)
        #bs x 36 x embed --> bs x 1 x 36 x embed
        features = features.unsqueeze(1)
        box_type_dim = box_type.size()
        #bs x 1 x 36 x embed --> bs x num_of_prop x 36 x embed
        features = features.repeat(1,box_type_dim[1],1,1)
        # bs x num_of_prop x 36 --> bs*num_of_prop x 36
        box_type = box_type.view(box_type_dim[0]*box_type_dim[1], box_type_dim[2])
        # bs*num_of_prop x 36 --> bs*num_of_prop x 36 x embeddim
        box_type_embed = self.embedding(box_type)
        # bs*num_of_prop x 36 x embed --> bs x num_of_prop x 36 x embed
        box_type_embed = box_type_embed.view(box_type_dim[0], box_type_dim[1], box_type_dim[2], -1)

        img_srl_emd = torch.cat([features, box_type_embed], dim=-1)
        img_srl_emd = img_srl_emd.view(img_srl_emd.size(0)*img_srl_emd.size(1),img_srl_emd.size(2), -1)
        img_srl_emd = self.fc2(img_srl_emd)

        img_srl_emd = self.dropout(img_srl_emd)

        #if self.data_name != 'f30k_precomp':
            #img_srl_emd = l2norm(img_srl_emd)

        # GCN reasoning
        # -> B,D,N
        GCN_img_emd = img_srl_emd.permute(0, 2, 1)
        GCN_img_emd = self.Rs_GCN_1(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_2(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_3(GCN_img_emd)
        GCN_img_emd = self.Rs_GCN_4(GCN_img_emd)
        # -> B,N,D
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)

        GCN_img_emd = l2norm(GCN_img_emd)

        rnn_img, hidden_state = self.img_rnn(GCN_img_emd)

        # features = torch.mean(rnn_img,dim=1)
        #print("image encoder: rnn_img {},  hidden_st:{}".format(rnn_img.size(), hidden_state.size()))
        #features = hidden_state[0]
        features = rnn_img.permute(0,2,1) #bs x dim x 36

        #if self.data_name == 'f30k_precomp':
        features = self.bn(features)

        features = features.permute(0,2,1) #bs x 36 x dim

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features, GCN_img_emd

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecompAttn, self).load_state_dict(new_state)

class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, srl_vocab_size, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size//2)

        self.embedding = nn.Embedding(srl_vocab_size, embed_size//2)
        self.fc2 = nn.Linear(embed_size, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)
        self.fc2.weight.data.uniform_(-r, r)
        self.fc2.bias.data.fill_(0)
    def forward(self, images, boxes, box_type, no_of_prop):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)
        #bs x 36 x embed --> bs x 1 x 36 x embed
        features = features.unsqueeze(1)
        box_type_dim = box_type.size()
        #bs x 1 x 36 x embed --> bs x num_of_prop x 36 x embed
        features = features.repeat(1,box_type_dim[1],1,1)
        # bs x num_of_prop x 36 --> bs*num_of_prop x 36
        box_type = box_type.view(box_type_dim[0]*box_type_dim[1], box_type_dim[2])
        # bs*num_of_prop x 36 --> bs*num_of_prop x 36 x embeddim
        box_type_embed = self.embedding(box_type)
        # bs*num_of_prop x 36 x embed --> bs x num_of_prop x 36 x embed
        box_type_embed = box_type_embed.view(box_type_dim[0], box_type_dim[1], box_type_dim[2], -1)

        img_srl_emd = torch.cat([features, box_type_embed], dim=-1)
        img_srl_emd = img_srl_emd.view(img_srl_emd.size(0)*img_srl_emd.size(1),img_srl_emd.size(2), -1)
        img_srl_emd = self.fc2(img_srl_emd)

        # normalize in the joint embedding space

        if not self.no_imgnorm:
            img_srl_emd = l2norm(img_srl_emd, dim=-1)

        # bs*num_of_prop x 36 x embed --> bs x num_of_prop x 36 x embed
        #img_srl_emd =img_srl_emd.view(box_type_dim[0], box_type_dim[1],box_type_dim[2], -1)

        return img_srl_emd

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        features = self.fc(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)


# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, srl_vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim//2)
        self.sembed = nn.Embedding(srl_vocab_size,word_dim//2)
        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.sembed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, srl, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        srl_size = srl.size()
        #bs x num_of_prop x seq_len --> bs*num_of_prop x seq_len
        srl = srl.view(srl_size[0]*srl_size[1],-1)
        y = self.sembed(srl)
        #bs*num_of_prop x seq_len x esize//2 ---> bs x num_of_prop x seq_len x esize//2
        y = y.view(srl_size[0],srl_size[1],srl_size[2],-1)
        x = x.unsqueeze(1)
        #bs x 1 x seq_len x esize//2 --> bs x num_of_prop x seq_len x esize//2
        x = x.repeat(1,srl_size[1],1,1 )
        #bs x num_of_prop x seq_len x esize
        embd = torch.cat([x,y], dim=-1)
        #bs x num_of_prop x seq_len x esize --> bs*num_of_prop x seq_len x esize
        embd = embd.view(embd.size(0)*embd.size(1),srl_size[2],-1 )


        #print("len==={}".format(lengths.size()))
        lengths_upd=[]
        for i in lengths:
            for j in range(srl_size[1]):
                lengths_upd.append(i)


        packed = pack_padded_sequence(embd,lengths_upd, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        if self.use_bi_gru:
            cap_emb = (cap_emb[:,:,:cap_emb.size(2)//2] + cap_emb[:,:,cap_emb.size(2)//2:])/2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        #cap_emb = cap_emb.view(srl_size[0],srl_size[1],srl_size[2],-1)
        return cap_emb, cap_len


def func_attention(query, context, opt, eps=1e-8):
    """
    query: (batch, queryL, d)
    context: (batch, sourceL, d)
    opt: parameters
    """
    batch_size, queryL, sourceL = context.size(
        0), query.size(1), context.size(1)

    # Step 1: preassign attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    attn = torch.bmm(context, queryT)
    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax(dim=1)(attn*opt.lambda_softmax)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)

    # Step 2: identify irrelevant fragments
    # Learning an indicator function H, one for relevant, zero for irrelevant
    if opt.focal_type == 'equal':
        funcH = focal_equal(attn, batch_size, queryL, sourceL)
    elif opt.focal_type == 'prob':
        funcH = focal_prob(attn, batch_size, queryL, sourceL)
    else:
        raise ValueError("unknown focal attention type:", opt.focal_type)

    # Step 3: reassign attention
    tmp_attn = funcH * attn
    attn_sum = torch.sum(tmp_attn, dim=-1, keepdim=True)
    re_attn = tmp_attn / attn_sum

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # --> (batch, sourceL, queryL)
    re_attnT = torch.transpose(re_attn, 1, 2).contiguous()
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, re_attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)
    funcH = torch.transpose(funcH, 1, 2).contiguous()

    return weightedContext, re_attnT#funcH#re_attnT


def focal_equal(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as equal
    sigma_{j} (xi - xj) = sigma_{j} xi - sigma_{j} xj
    attn: (batch, queryL, sourceL)
    """
    funcF = attn * sourceL - torch.sum(attn, dim=-1, keepdim=True)
    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn


def focal_prob(attn, batch_size, queryL, sourceL):
    """
    consider the confidence g(x) for each fragment as the sqrt
    of their similarity probability to the query fragment
    sigma_{j} (xi - xj)gj = sigma_{j} xi*gj - sigma_{j} xj*gj
    attn: (batch, queryL, sourceL)
    """

    # -> (batch, queryL, sourceL, 1)
    xi = attn.unsqueeze(-1).contiguous()
    # -> (batch, queryL, 1, sourceL)
    xj = attn.unsqueeze(2).contiguous()
    # -> (batch, queryL, 1, sourceL)
    xj_confi = torch.sqrt(xj)

    xi = xi.view(batch_size*queryL, sourceL, 1)
    xj = xj.view(batch_size*queryL, 1, sourceL)
    xj_confi = xj_confi.view(batch_size*queryL, 1, sourceL)

    # -> (batch*queryL, sourceL, sourceL)
    term1 = torch.bmm(xi, xj_confi)
    term2 = xj * xj_confi
    funcF = torch.sum(term1-term2, dim=-1)  # -> (batch*queryL, sourceL)
    funcF = funcF.view(batch_size, queryL, sourceL)

    fattn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return fattn

def manipulate_attention_guide(attention_guide, cross_attn, opt, eps=1e-8):
    '''
    attention_guide: bathc x 36 x seq_len
    '''
    attn = attention_guide
    batch_size = attention_guide.size(0)
    smooth = opt.lambda_softmax
    if cross_attn == 'i2t':
        #b x seq_len x 36
        attn = attn.permute(0,2,1)#make it (b,sourceL,queryL)

    sourceL = attn.size(1)
    queryL = attn.size(2)

    attn = nn.LeakyReLU(0.1)(attn)
    attn = l2norm(attn, 2)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax()(attn)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    #sum is one in sourCeL dimenstion
    return attnT

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def xattn_score_t2i(images, captions, cap_lens, opt, mode = 'train'):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    attentions = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    max_len = max(cap_lens)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)
        #print(attn[i].size())
        if mode == 'train':
            attn_i = torch.zeros(attn[i].size(0), max_len, dtype = attn[i].dtype)
            attn_i[:,:attn[i].size(1)] = attn[i][:,:attn[i].size(1)]
        #print(attn_i.size())
            attentions.append(attn_i) # assuming ith image and ith caption pairs

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    # (batch, 36, num_tok) #assume n_images ==n_caption. May not be true. Deal later
    #attentions = None
    if mode == 'train':
        attentions = torch.stack(attentions,0).cuda()
    return similarities, attentions


def xattn_score_i2t(images, captions, cap_lens, opt, mode = 'train'):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    attentions = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    max_len = max(cap_lens)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)
        if mode == 'train':
            attn_i = torch.zeros(max_len,attn[i].size(1), dtype = attn[i].dtype)
            attn_i[:attn[i].size(0),:] = attn[i][:attn[i].size(0),:]
            attentions.append(attn_i) # assuming ith image and ith caption pairs
        #attentions.append(attn[i])# assuming ith image and ith caption pairs

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)

    # (batch,  num_tok, 36) #assume n_images ==n_caption. May not be true. Deal later
    if mode == 'train':
        attentions = torch.stack(attentions,0).cuda()
    return similarities, attentions

def xattn_score(images, captions, cap_lens, opt, mode = 'train'):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    max_len = max(cap_lens)
    attentions_t2i = []
    attentions_i2t = []
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)

        # Focal attention in text-to-image direction
        # weiContext: (n_image, n_word, d)
        weiContext, attnT_t2i = func_attention(cap_i_expand, images, opt)
        t2i_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)


        # Focal attention in image-to-text direction
        # weiContext: (n_image, n_word, d)
        weiContext, attnT_i2t= func_attention(images, cap_i_expand, opt)
        i2t_sim = cosine_similarity(images, weiContext, dim=2)


        if opt.agg_func == 'LogSumExp':
            t2i_sim.mul_(opt.lambda_lse).exp_()
            t2i_sim  = t2i_sim.sum(dim=1, keepdim=True)
            t2i_sim = torch.log(t2i_sim)/opt.lambda_lse

            i2t_sim.mul_(opt.lambda_lse).exp_()
            i2t_sim = i2t_sim.sum(dim=1, keepdim=True)
            i2t_sim = torch.log(i2t_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            t2i_sim  = t2i_sim.max(dim=1, keepdim=True)[0]
            i2t_sim = i2t_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            t2i_sim  = t2i_sim.sum(dim=1, keepdim=True)
            i2t_sim = i2t_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            t2i_sim = t2i_sim.mean(dim=1, keepdim=True)
            i2t_sim = i2t_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))

        # Overall similarity for image and text
        sim = t2i_sim + i2t_sim
        similarities.append(sim)
        if mode == 'train':
            attn_i = torch.zeros(max_len,attnT_i2t[i].size(1), dtype = attnT_i2t[i].dtype)
            #print("attn_i {}".format(attn_i.size()))
            #print("attnT_i2t {}".format(attnT_i2t[i].size()))
            attn_i[:attnT_i2t[i].size(0),:] = attnT_i2t[i][:attnT_i2t[i].size(0),:]
            attentions_i2t.append(attn_i) # assuming ith image and ith caption pairs


            attn_t = torch.zeros(attnT_t2i[i].size(0), max_len, dtype = attnT_t2i[i].dtype)
            #print("attn_t {}".format(attn_t.size()))
            #print("attn_t2i {}".format(attnT_t2i[i].size()))
            attn_t[:,:attnT_t2i[i].size(1)] = attnT_t2i[i][:,:attnT_t2i[i].size(1)]
            attentions_t2i.append(attn_t)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    if mode == 'train':
        attentions_t2i = torch.stack(attentions_t2i,0).cuda()
        attentions_i2t = torch.stack(attentions_i2t,0).cuda()

    return similarities, attentions_t2i, attentions_i2t
    #return similarities, attnT_t2i, attnT_i2t


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l):
        # compute image-sentence score matrix
        if self.opt.cross_attn == 't2i':
            scores, attentions_t2i = xattn_score_t2i(im, s, s_l, self.opt)
            attentions_i2t  = None
        elif self.opt.cross_attn == 'i2t':
            scores, attentions_i2t  = xattn_score_i2t(im, s, s_l, self.opt)
            attentions_t2i = None
        elif self.opt.cross_attn == 'both':
            scores, attentions_t2i,  attentions_i2t = xattn_score(im, s, s_l, self.opt)
        else:
            raise ValueError("unknown first norm type:", opt.raw_feature_norm)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]


        return cost_s.sum() + cost_im.sum(), attentions_t2i,  attentions_i2t


class SCAN(object):
    """
    Stacked Cross Attention Network (SCAN) model
    """
    def __init__(self, opt):
        # Build Models
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.srl_vocab_size, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt.w_vocab_size, opt.bio_vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_bi_gru=opt.bi_gru,
                                   no_txtnorm=opt.no_txtnorm)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)

        self.KL = JsdCrossEntropy()#torch.nn.KLDivLoss(reduction = 'batchmean').cuda()
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, boxes, box_type, captions, lengths,no_of_prop, w_type_padded, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            box_type = box_type.cuda()
            w_type_padded = w_type_padded.cuda()

            #print(images.size())
            #print(captions.size())

        # Forward
        img_emb = self.img_enc(images,boxes, box_type, no_of_prop)

        # cap_emb (tensor), cap_lens (list)
        cap_emb, cap_lens = self.txt_enc(captions,w_type_padded, lengths)
        
        if self.opt.precomp_enc_type == "GCN":
            img_emb = img_emb[1]#GCN_emd

        return img_emb, cap_emb, cap_lens


    def forward_loss(self, img_emb, cap_emb, cap_len, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss, attention_t2i, attention_i2t  = self.criterion(img_emb, cap_emb, cap_len)
        #self.logger.update('Le', loss.item, img_emb.size(0))
        return loss, attention_t2i, attention_i2t

    def train_emb(self,images, boxes, box_type, captions, lengths, ids, no_of_prop,w_type_padded,*args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens = self.forward_emb(images, boxes, box_type, captions, lengths, no_of_prop, w_type_padded)
        #print(cap_lens)
        #print("==img=={}, cap=={} caplen=={}".format(img_emb.size(), cap_emb.size(),cap_lens.size()))
        # measure accuracy and record loss

        #print(lengths)
        #print('---------------------------------')
        self.optimizer.zero_grad()
        loss, attention_t2i, attention_i2t  = self.forward_loss(img_emb, cap_emb, cap_lens)


        self.logger.update('Loss', loss.item(), img_emb.size(0))
        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
