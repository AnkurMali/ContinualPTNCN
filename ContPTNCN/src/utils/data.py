import random
import tensorflow as tf
import numpy as np
import io
import sys
import math
seed = 1234
tf.random.set_seed(seed=seed)
np.random.seed(seed)

PAD, UNK, CLS, SEP, MASK = '<-PAD->', '<-UNK->', '<-CLS->', '<-SEP->', '<-MASK->'

class Vocab(object):
    def __init__(self, filename, min_occur_cnt=-1, specials=None, use_file_only=True):
        if use_file_only:
            idx2token = []
        else:
            idx2token = [PAD, UNK] + (specials if specials is not None else [])
        self._token2idx = {}
        idx = 0
        for line in open(filename, encoding='utf8').readlines():
            try:
                datum = line.strip().split() #token, cnt = line.strip().split()
                cnt = 0
                if len(datum) > 1:
                    token = datum[0]
                    cnt = datum[1]
                else:
                    token = datum[0]
            except:
                continue
            if int(cnt) >= min_occur_cnt:
                idx2token.append(token)
                #self._token2idx.update({token[0] : idx})
                idx += 1

        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        #print((self._token2idx))
        self._idx2token = idx2token
        if specials is not None:
            self._padding_idx = self._token2idx[PAD]
            self._unk_idx = self._token2idx[UNK]
        else:
            self._unk_idx = -1
            self._padding_idx = -1

    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def random_token(self):
        return self.idx2token(1 + np.random.randint(self.size - 1))

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)

# class Vocab(object):
# 	def __init__(self, filename, min_occur_cnt=2, specials=None):
# 		idx2token = [PAD, UNK] + (specials if specials is not None else [])
# 		cnt = 0
# 		for line in open(filename, encoding='utf8').readlines():
#             try:
#                 token = line.strip().split()
#             except:
#                 continue
#             if int(cnt) >= min_occur_cnt:
#                 idx2token.append(token)
# 		#print(" >> Tokens read in from vocab file:  ",cnt)
#         self._token2idx = dict(zip(idx2token, range(len(idx2token))))
#         self._idx2token = idx2token
# 		self._padding_idx = self._token2idx[PAD]
# 		self._unk_idx = self._token2idx[UNK]
# 		#print(" >> Total Vocab.Size = ",self.size)
#
# 	@property
# 	def size(self):
# 		return len(self._idx2token)
#
# 	@property
# 	def unk_idx(self):
# 		return self._unk_idx
#
# 	@property
# 	def padding_idx(self):
# 		return self._padding_idx
#
# 	def random_token(self):
# 		return self.idx2token(1 + np.random.randint(self.size - 1))
#
# 	def idx2token(self, x):
# 		if isinstance(x, list):
# 		    return [self.idx2token(i) for i in x]
# 		return self._idx2token[x]
#
# 	def token2idx(self, x):
# 		if isinstance(x, list):
# 		    return [self.token2idx(i) for i in x]
#
#
# 		return self._token2idx.get(x, self.unk_idx)
