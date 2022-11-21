import pickle
import pandas as pd
import numpy as np
import scipy.sparse as sp
from collections import Counter
from tqdm import tqdm


def _calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def _calculate_random_walk_laplacian(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


class WordVocab:
    def __init__(self, roadmap_path, traj_path, min_freq=1, use_mask=True, seq_len=128, eos=False):
        self.pad_index = 0
        self.unk_index = 1
        self.sos_index = 2  # CLS
        if use_mask:
            self.mask_index = 3
            specials = ["<pad>", "<unk>", "<sos>", "<mask>"]
            if eos:
                self.sep_index = 4
                specials = ["<pad>", "<unk>", "<sos>", "<mask>", "<eos>"]
        else:
            specials = ["<pad>", "<unk>", "<sos>"]

        train = pd.read_csv(traj_path, sep=';')

        counter = Counter()
        paths = train['path'].values
        for i in tqdm(range(paths.shape[0]), desc='location counting'):
            path_l = eval(paths[i])[:seq_len]
            for p in path_l:
                counter[p] += 1
        self.freqs = counter
        min_freq = max(min_freq, 1)

        self.index2loc = list(specials)
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        self.del_edge = []
        self.all_edge = []
        for word, freq in words_and_frequencies:
            if freq < min_freq:
                self.del_edge.append(word)
                continue
            self.index2loc.append(word)
            self.all_edge.append(word)
        self.loc2index = {tok: i for i, tok in enumerate(self.index2loc)}
        self.vocab_size = len(self.index2loc)

        users = train['usr_id'].values
        users = np.unique(users)
        specials = ["<pad>", "<unk>"]
        self.index2usr = list(specials)
        for u in users:
            self.index2usr.append(u)
        self.usr2index = {tok: i for i, tok in enumerate(self.index2usr)}
        self.user_num = len(self.index2usr)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.loc2index.get(word, self.unk_index) for word in sentence]

        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.index2loc[idx]
                 if idx < len(self.index2loc)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'WordVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)

    def __eq__(self, other):
        if self.loc2index != other.loc2index:
            return False
        if self.index2loc != other.index2loc:
            return False
        return True

    def __len__(self):
        return len(self.index2loc)

    def vocab_rerank(self):
        self.loc2index = {word: i for i, word in enumerate(self.index2loc)}

    def extend(self, v, sort=False):
        words = sorted(v.index2loc) if sort else v.index2loc
        for w in words:
            if w not in self.loc2index:
                self.index2loc.append(w)
                self.loc2index[w] = len(self.index2loc) - 1
