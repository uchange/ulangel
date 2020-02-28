import html
import re

import emoji

UNK = "xxunk"
PAD = "xxpad"
BOS = "xxbos"
EOS = "xxeos"
TK_REP = "xxrep"
TK_WREP = "xxwrep"
TK_UP = "xxup"
TK_MAJ = "xxmaj"
BOS = "xbos"  # beginning-of-sentence tag
FLD = "xfld"


def toLowercase(matchobj):
    return matchobj.group(1).lower()


def lower_tw(tweet):
    tweet_with_lower_tw = re.sub(r"(@\S+){1}", toLowercase, tweet)
    return tweet_with_lower_tw


def fixup(x):
    re1 = re.compile(r"  +")
    x = (
        x.replace("#39;", "'")
        .replace("&amp;", "&")
        .replace("#146;", "'")
        .replace("nbsp;", " ")
        .replace("#36;", "$")
        .replace("\\n", "\n")
        .replace("quot;", "'")
        .replace("<br />", "\n")
        .replace('\\"', '"')
        .replace("<unk>", "u_n")
        .replace(" @.@ ", ".")
        .replace(" @-@ ", "-")
        .replace("\\", " \\ ")
        .replace("\xa0", "")
    )
    x = re.sub(r"[h][t][t][p]\S+", "", x)
    x = emoji.demojize(x)
    return re1.sub(" ", html.unescape(x))


def spec_add_spaces(x):
    "Add spaces around / and #"
    x = re.sub(r"([/#@])", r" \1 ", x)
    x = re.sub(r"([:][\S]+[:])", r" \1 ", x.replace("::", ": :"))
    return x


def rm_useless_spaces(x):
    "Remove multiple spaces"
    return re.sub(" {2,}", " ", x)


def replace_all_caps(x):
    "Replace tokens in ALL CAPS by their lower version and add `TK_UP` ahead."
    res = []
    for t in x:
        if t.isupper() and len(t) > 1:
            res.append(TK_UP)
            res.append(t.lower())
        else:
            res.append(t)
    return res


def deal_caps(x):
    "Replace all Capitalized tokens in by their lower version and add `TK_MAJ` ahead."
    res = []
    for t in x:
        if t == "":
            continue
        if t[0].isupper() and len(t) > 1 and t[1:].islower():
            res.append(TK_MAJ)
        res.append(t.lower())
    return res


def replace_rep(x):
    "Replace repetitions at the character level: cccc -> TK_REP 4 c"

    def _replace_rep(m):
        c, cc = m.groups()
        return f" {TK_REP} {len(cc)+1} {c} "

    re_rep = re.compile(r"(\S)(\1{3,})")
    return re_rep.sub(_replace_rep, x)


def replace_wrep(x):
    "Replace word repetitions: word word word -> TK_WREP 3 word"

    def _replace_wrep(m):
        c, cc = m.groups()
        return f" {TK_WREP} {len(cc.split())+1} {c} "

    re_wrep = re.compile(r"(\b\w+\W+)(\1{3,})")
    return re_wrep.sub(_replace_wrep, x)


def text_proc(txts, tokenizer):
    # remove charaters that we don't want
    texts = fixup(txts)
    # replace repeted word
    texts = replace_wrep(texts)
    # replace repeted letter
    texts = replace_rep(texts)
    # add spaces before or after / # @ symbols
    texts = spec_add_spaces(texts)
    # remove useless spaces
    texts = rm_useless_spaces(texts)
    # tokenize the text
    tok = [t.text for t in tokenizer(texts)]
    # replace all words in capital
    tok = replace_all_caps(tok)
    # replace words begining with a capital letter
    tok = deal_caps(tok)
    tok = [BOS] + tok + [EOS]
    return [tok]
