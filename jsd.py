import torch
import torch.nn as nn
import torch.nn.functional as F

#from .cross_entropy import LabelSmoothingCrossEntropy
class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class JsdCrossEntropy(nn.Module):
    """ Jensen-Shannon Divergence + Cross-Entropy Loss
    Based on impl here: https://github.com/google-research/augmix/blob/master/imagenet.py
    From paper: 'AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty -
    https://arxiv.org/abs/1912.02781
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/loss/jsd.py
    Hacked together by / Copyright 2020 Ross Wightman
    """
    def __init__(self,alpha=12, smoothing=0.1):
        super().__init__()

        self.alpha = alpha
        if smoothing is not None and smoothing > 0:
            self.cross_entropy_loss = LabelSmoothingCrossEntropy(smoothing)
        else:
            self.cross_entropy_loss = torch.nn.CrossEntropyLoss()

    def forward(self, output, target):

        logits_split = [output, target]
        # Cross-entropy is only computed on clean images
        #loss = self.cross_entropy_loss(output, target)
        probs = [F.softmax(logits, dim=-1) for logits in logits_split]

        # Clamp mixture distribution to avoid exploding KL divergence
        logp_mixture = torch.clamp(torch.stack(probs).mean(axis=0), 1e-7, 1).log()
        loss = self.alpha * sum([F.kl_div(
            logp_mixture, p_split, reduction='batchmean') for p_split in probs]) / len(probs)
        return loss
