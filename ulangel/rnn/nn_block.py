import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ulangel.rnn.dropouts import (
    ActivationDropout,
    ConnectionWeightDropout,
    EmbeddingDropout,
)


class AWD_LSTM(nn.Module):
    """AWD-LSTM inspired by https://arxiv.org/pdf/1708.02182.pdf.
    LSTM with embedding dropout, connection weight dropout, activation dropout.
    """

    initrange = 0.1

    def __init__(
        self,
        vocab_sz,
        emb_sz,
        n_hid,
        n_layers,
        pad_token,
        hidden_p=0.2,
        input_p=0.6,
        embed_p=0.1,
        weight_p=0.5,
    ):
        super().__init__()
        self.bs = 1
        self.emb_sz = emb_sz
        self.n_hid = n_hid
        self.n_layers = n_layers
        self.pad_token = pad_token
        self.emb = nn.Embedding(vocab_sz, emb_sz, padding_idx=pad_token)
        self.emb_dp = EmbeddingDropout(self.emb, embed_p)
        self.rnns = [
            nn.LSTM(
                emb_sz if l == 0 else n_hid,
                (n_hid if l != n_layers - 1 else emb_sz),
                1,
                batch_first=True,
            )
            for l in range(n_layers)
        ]
        self.rnns = nn.ModuleList(
            [ConnectionWeightDropout(rnn, weight_p) for rnn in self.rnns]
        )
        self.emb.weight.data.uniform_(-self.initrange, self.initrange)
        self.input_dp = ActivationDropout(input_p)
        self.hidden_dps = nn.ModuleList(
            [ActivationDropout(hidden_p) for l in range(n_layers)]
        )

    def forward(self, input):
        input = input.long()
        bs, sl = input.size()
        if bs != self.bs:
            self.bs = bs
            self.reset()

        mask = input == self.pad_token
        lengths = sl - mask.long().sum(1)
        n_empty = (lengths == 0).sum()
        if n_empty > 0:
            input = input[:-n_empty]
            lengths = lengths[:-n_empty]
            self.hidden = [
                (h[0][:, : input.size(0)], h[1][:, : input.size(0)])
                for h in self.hidden
            ]
        raw_output = self.input_dp(self.emb_dp(input))
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, (rnn, hid_dp) in enumerate(zip(self.rnns, self.hidden_dps)):
            raw_output = pack_padded_sequence(raw_output, lengths, batch_first=True)
            raw_output, new_h = rnn(raw_output, self.hidden[l])
            raw_output = pad_packed_sequence(raw_output, batch_first=True)[0]
            raw_outputs.append(raw_output)
            if l != self.n_layers - 1:
                raw_output = hid_dp(raw_output)
            outputs.append(raw_output)
            new_hidden.append(new_h)

        self.hidden = self.to_detach(new_hidden)
        return raw_outputs, outputs, mask

    def to_detach(self, h):
        "Detaches `h` from its history."
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(self.to_detach(v) for v in h)

    #         return h.detach() if type(h) == torch.Tensor else tuple(to_detach(v) for v in h)

    def _one_hidden(self, l):
        "Return one hidden state."
        nh = self.n_hid if l != self.n_layers - 1 else self.emb_sz
        return next(self.parameters()).new(1, self.bs, nh).zero_()

    def reset(self):
        "Reset the hidden states."
        self.hidden = [
            (self._one_hidden(l), self._one_hidden(l)) for l in range(self.n_layers)
        ]


class LinearDecoder(nn.Module):
    """The inverse of the embedding layer. Transform embedding vectors back to
    its corresponding integers.
    """

    def __init__(self, n_out, n_hid, output_p, tie_encoder=None, bias=True):
        super().__init__()
        self.output_dp = ActivationDropout(output_p)
        self.decoder = nn.Linear(n_hid, n_out, bias=bias)
        if bias:
            self.decoder.bias.data.zero_()
        if tie_encoder:
            self.decoder.weight = tie_encoder.weight
        else:
            init.kaiming_uniform_(self.decoder.weight)

    def forward(self, input):
        raw_outputs, outputs, mask = input
        output = self.output_dp(outputs[-1]).contiguous()
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        return decoded, raw_outputs, outputs


class SequentialRNN(nn.Sequential):
    """A sequential module that passes the reset call to its children."""

    def reset(self):
        for c in self.children():
            if hasattr(c, "reset"):
                c.reset()


class TextOnlySentenceEncoder(nn.Module):
    """The same as the language model encoder, but if the input texts are
    longer than the bptt, it cuts them into bptt in order to be calculated by
    the language model, and then concatenate the results to make still one text
    represented by one tensor
    """

    def __init__(self, module, bptt, pad_idx=1):
        super().__init__()
        self.bptt = bptt
        self.module = module
        self.pad_idx = pad_idx

    def concat(self, arrs, bs):
        return [
            torch.cat([self.pad_tensor(l[si], bs) for l in arrs], dim=1)
            for si in range(len(arrs[0]))
        ]

    def pad_tensor(self, t, bs, val=0.0):
        "for a batch which size is smaller than a batchsize, fill in with zeros to make it a batchsize"
        if t.size(0) < bs:
            return torch.cat([t, val + t.new_zeros(bs - t.size(0), *t.shape[1:])])
        return t

    def forward(self, input):
        input = input.long()
        bs, sl = input.size()
        self.module.reset()
        raw_outputs = []
        outputs = []
        masks = []
        for i in range(0, sl, self.bptt):
            r, o, m = self.module(input[:, i : min(i + self.bptt, sl)])
            masks.append(self.pad_tensor(m, bs, 1))
            raw_outputs.append(r)
            outputs.append(o)
        return (
            self.concat(raw_outputs, bs),
            self.concat(outputs, bs),
            torch.cat(masks, dim=1),
        )


class TextPlusSentenceEncoder(nn.Module):
    def __init__(self, module, bptt, pad_idx=1):
        super().__init__()
        self.bptt = bptt
        self.module = module
        self.pad_idx = pad_idx

    def concat(self, arrs, bs):
        return [
            torch.cat([self.pad_tensor(l[si], bs) for l in arrs], dim=1)
            for si in range(len(arrs[0]))
        ]

    def pad_tensor(self, t, bs, val=0.0):
        "for a batch which size is smaller than a batchsize, fill in with zeros to make it a batchsize"
        if t.size(0) < bs:
            return torch.cat([t, val + t.new_zeros(bs - t.size(0), *t.shape[1:])])
        return t

    def forward(self, input):
        if len(input) != 2:
            ids_input, kw_ls = list(zip(*input))
            ids_input = torch.stack(ids_input)
            kw_ls = torch.stack(kw_ls)
        else:
            ids_input, kw_ls = input
        bs, sl = ids_input.size()
        self.module.reset()
        raw_outputs = []
        outputs = []
        masks = []
        for i in range(0, sl, self.bptt):
            ids_input_i = ids_input[:, i : min(i + self.bptt, sl)]
            r, o, m = self.module(ids_input_i)
            masks.append(self.pad_tensor(m, bs, 1))
            raw_outputs.append(r)
            outputs.append(o)
        return (
            self.concat(raw_outputs, bs),
            self.concat(outputs, bs),
            torch.cat(masks, dim=1),
            kw_ls,
        )


class TextOnlyPoolingLinearClassifier(nn.Module):
    """Create a linear classifier with pooling. Concatenating the last sequence
    of outputs, the max pooling of outputs, the average pooling of outputs. This
    concatenation is the input of the lineal neuro network classifier.
    """

    def __init__(self, layers, drops):
        super().__init__()
        mod_layers = []
        activs = [nn.ReLU(inplace=True)] * (len(layers) - 1)
        for n_in, n_out, p, actn in zip(layers[:-1], layers[1:], drops, activs):
            mod_layers += self.bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers = nn.Sequential(*mod_layers)

    def forward(self, input):
        raw_outputs, outputs, mask = input
        output = outputs[-1]
        lengths = output.size(1) - mask.long().sum(dim=1)
        avg_pool = output.masked_fill(mask[:, :, None], 0).sum(dim=1)
        avg_pool.div_(lengths.type(avg_pool.dtype)[:, None])
        max_pool = output.masked_fill(mask[:, :, None], -float("inf")).max(dim=1)[0]
        x = torch.cat(
            [output[torch.arange(0, output.size(0)), lengths - 1], max_pool, avg_pool],
            1,
        )  # Concat pooling.
        x = self.layers(x)
        return x

    def bn_drop_lin(self, n_in, n_out, bn=True, p=0.0, actn=None):
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0:
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None:
            layers.append(actn)
        return layers


class TextPlusPoolingLinearClassifier(nn.Module):
    "Create a linear classifier with pooling."

    def __init__(self, layers1, drops1, layers2, drops2):
        super().__init__()
        mod_layers1 = []
        activs1 = [nn.ReLU(inplace=True)] * (len(layers1) - 1)
        for n_in, n_out, p, actn in zip(layers1[:-1], layers1[1:], drops1, activs1):
            mod_layers1 += self.bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers1 = nn.Sequential(*mod_layers1)

        mod_layers2 = []
        activs2 = [nn.ReLU(inplace=True)] * (len(layers2) - 1)
        for n_in, n_out, p, actn in zip(layers2[:-1], layers2[1:], drops2, activs2):
            mod_layers2 += self.bn_drop_lin(n_in, n_out, p=p, actn=actn)
        self.layers2 = nn.Sequential(*mod_layers2)

    def forward(self, input):
        raw_outputs, outputs, mask, kw_ls = input
        output = outputs[-1]
        lengths = output.size(1) - mask.long().sum(dim=1)
        avg_pool = output.masked_fill(mask[:, :, None], 0).sum(dim=1)
        avg_pool.div_(lengths.type(avg_pool.dtype)[:, None])
        max_pool = output.masked_fill(mask[:, :, None], -float("inf")).max(dim=1)[0]
        x1 = torch.cat(
            [output[torch.arange(0, output.size(0)), lengths - 1], max_pool, avg_pool],
            1,
        )  # Concat pooling.
        first_clas = self.layers1(x1)
        x2 = torch.cat([first_clas, kw_ls], 1)
        x = self.layers2(x2)
        return x

    def bn_drop_lin(self, n_in, n_out, bn=True, p=0.0, actn=None):
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0:
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None:
            layers.append(actn)
        return layers
