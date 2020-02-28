import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class LanguageModelDataset:
    """LanguageModelDataset is like a dataset for classification data returning
    tuples of x and y. While for a language model, y is a tensor of words
    shifted one step after x.
    It shuffles the order of all texts is needed, concatenates all texts end to
    end, and then batchify the corpus into [batchsize, nomber of batches]. At
    each iteration, it returns a tuple of texte x and texte y."""

    def __init__(self, data, bs=64, bptt=70, shuffle=False):
        self.data = data
        self.bs = bs
        self.bptt = bptt
        self.shuffle = shuffle
        total_len = sum([len(t) for t in data])
        self.n_batch = total_len // bs
        self._batchify()

    def __len__(self):
        return ((self.n_batch - 1) // self.bptt) * self.bs

    def __getitem__(self, idx):
        source = self.batched_data[idx % self.bs]
        seq_idx = (idx // self.bs) * self.bptt
        return (
            source[seq_idx : seq_idx + self.bptt],
            source[seq_idx + 1 : seq_idx + self.bptt + 1],
        )

    def _batchify(self):
        texts = self.data
        if self.shuffle:
            texts = texts[torch.randperm(len(texts))]
        stream = torch.cat([torch.Tensor(t) for t in texts])
        self.batched_data = stream[: self.n_batch * self.bs].view(self.bs, self.n_batch)


class DataBunch:
    """DataBunch is a class that gathers training and validation DataLoader
    (DataLoader of pytorch returns a batch of dataset(batchsize tuples of x and
    y at each iteration)).
    """

    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl = train_dl
        self.valid_dl = valid_dl
        self.c = c

    @property
    def train_ds(self):
        return self.train_dl.dataset

    @property
    def valid_ds(self):
        return self.valid_dl.dataset


class TextClassificationDataset(Dataset):
    """TextClassificationDataset is a dataset for text classification data
    returning a tuple of x and y at each iteration.
    x is an array of indexes of the text, y is an integer corresponding to the
    class label.
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        x = self.x[idx]
        return np.array(x), self.y[idx]

    def __len__(self):
        return len(self.x)


# validation sampler, just sort from the longest to the shortest
class ValidationSampler(Sampler):
    """ValidationSampler is a sampler for the validation data.
    It sorts the data in the way defined by the given key, in an increasing or
    a decreasing way.
    """

    def __init__(self, data_source, key):
        self.data_source = data_source
        self.key = key

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        return iter(
            sorted(list(range(len(self.data_source))), key=self.key, reverse=True)
        )


class TrainingSampler(Sampler):
    """TrainSampler is a sampler for the training data.
    It sorts the data in the way defined by the given key, the longest at the
    first, the shortest at the end, random in the middle.
    1. It randomizes the text data, and puts them into a megabatch which is 50
    times larger than a normal batch.
    2. Sorting all megabatches by the given key, in a decreasing order,
    concatenating the sorted result.
    3. Deviding data into normal batches, looking for the index of the longest
    batch
    4. Switching the position of the current first batch and the longest batch,
    to make sure that the longest batch is at the first position.
    """

    def __init__(self, data_source, key, bs):
        self.data_source = data_source
        self.key = key
        self.bs = bs

    def __len__(self):
        return len(self.data_source)

    def __iter__(self):
        idxs = torch.randperm(len(self.data_source))
        megabatches = [
            idxs[i : i + self.bs * 50] for i in range(0, len(idxs), self.bs * 50)
        ]
        sorted_idx = torch.cat(
            [torch.tensor(sorted(s, key=self.key, reverse=True)) for s in megabatches]
        )
        batches = [
            sorted_idx[i : i + self.bs] for i in range(0, len(sorted_idx), self.bs)
        ]
        # find the chunk with the largest key
        max_idx = torch.argmax(torch.tensor([self.key(ck[0]) for ck in batches]))
        batches[0], batches[max_idx] = batches[max_idx], batches[0]
        batch_idxs = torch.randperm(len(batches) - 2)
        sorted_idx = (
            torch.cat([batches[i + 1] for i in batch_idxs])
            if len(batches) > 1
            else torch.LongTensor([])
        )
        sorted_idx = torch.cat([batches[0], sorted_idx, batches[-1]])
        return iter(sorted_idx)


def pad_collate_textonly(samples, pad_idx=1, pad_first=False):
    """For each batch, fill in all texts with the pad_idx, in order to make all
    lines have the same length as the longest text of this batch.
    A sample is a batch of dataset (a batch of tuples of x and y). s[0] is x,
    s[1] is y.
    When pad_first=True, the text is at the end, when pad_first=False, the text
    is in the beginning.
    """
    max_len = max([len(s[0]) for s in samples])
    # res is a matrix
    res = torch.zeros(len(samples), max_len).long() + pad_idx
    for i, s in enumerate(samples):
        if pad_first:
            res[i, -len(s[0]) :] = torch.LongTensor(s[0])
        else:
            res[i, : len(s[0])] = torch.LongTensor(s[0])
    return res, torch.tensor([s[1] for s in samples])


def pad_collate_textplus(samples, pad_idx=1, pad_first=False):
    max_len = max([len(s[0][0]) for s in samples])
    x = []
    for i, s in enumerate(samples):
        xi = []
        # res is a list
        res = [pad_idx] * max_len
        x1, *x2 = s[0]
        if pad_first:
            res[-len(x1) :] = np.array(x1)
        else:
            res[: len(x1)] = np.array(x1)
        xi.append(torch.LongTensor(res))
        xi.append(torch.FloatTensor(x2))
        x.append(xi)
    return x, torch.tensor([s[1] for s in samples])
