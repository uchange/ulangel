import warnings

import torch.nn as nn
import torch.nn.functional as F


def dropout_mask(x, sz, p):
    """create a mask to zero out p persent of the activation, by keeping the
    same module of the tensor."""
    return x.new(*sz).bernoulli_(1 - p).div_(1 - p)


class ActivationDropout(nn.Module):
    """zeroing out p percent of the layer activation, returning a layer of
    actication with p persent of zeros.
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        size = (x.size(0), 1, x.size(2))
        m = dropout_mask(x.data, size, self.p)
        return x * m


class EmbeddingDropout(nn.Module):
    """Applies dropout on the embedding layer by zeroing out some elements of
    the embedding vector. It's the activation dropout of the embedding layers.
    Returning an embedding object.
    """

    def __init__(self, emb, embed_p):
        super().__init__()
        self.emb = emb
        self.embed_p = embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None:
            self.pad_idx = -1

    def forward(self, words, scale=None):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0), 1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        if scale:
            masked_embed.mul_(scale)
        return F.embedding(
            words,
            masked_embed,
            self.pad_idx,
            self.emb.max_norm,
            self.emb.norm_type,
            self.emb.scale_grad_by_freq,
            self.emb.sparse,
        )


class ConnectionWeightDropout(nn.Module):
    """zeroing out p percent of the connection weights between defined layers
    (hh in the default setting), returning a matrix of connection weights with
    p persent of zeros.
    """

    def __init__(self, module, weight_p=[0.0], layer_names=["weight_hh_l0"]):
        super().__init__()
        self.module = module
        self.weight_p = weight_p
        self.layer_names = layer_names
        for layer in self.layer_names:
            # for each layer weight Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            # save the original weights
            self.module.register_parameter(f"{layer}_raw", nn.Parameter(w.data))
            # apply dropout to this layer and replace the full weights
            self.module._parameters[layer] = F.dropout(
                w, p=self.weight_p, training=False
            )

    def _setweights(self):
        for layer in self.layer_names:
            raw_w = getattr(self.module, f"{layer}_raw")
            self.module._parameters[layer] = F.dropout(
                raw_w, p=self.weight_p, training=self.training
            )

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)
