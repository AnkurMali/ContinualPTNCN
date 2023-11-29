import random
#import tensorflow as tf
import numpy as np
import io
import sys
import math
from utils import shuffle_pair

seed = 1234
#tf.random.set_seed(seed=seed)
np.random.seed(seed)

'''
  A discrete variable sequence sampler -- note that this does dynamic padding
  to ensure all sequences in a mini-batch are of the same length (pad is a tokens
  index of -1).

  @author Ankur Mali
'''
class DataLoader(object):

    def __init__(self, xfname, batch_size):
        self.x_fname = xfname
        self.batch_size = batch_size

        self.xdata = []
        self.max_seq_len = 0
        fd = open(xfname, 'r')
        for line in fd:
            tok_seq = line.split(",")
            if len(tok_seq) > 0:
                seq = [int(elem) for elem in tok_seq]
                self.xdata.append(seq)
                self.max_seq_len = max(self.max_seq_len, len(seq))
            # else, discard this sequence since its NULL or empty
        fd.close()

        print("---------------------------------------------------------------")
        print("  Dataset: ",self.x_fname)
        print("  Len(xdata) = ", len(self.xdata))
        print("  Max_seq_len = ", self.max_seq_len)
        print("---------------------------------------------------------------")
        self.ptrs = None #np.random.permutation(len(self.xdata))

    def __iter__(self):
        self.ptrs = np.random.permutation(len(self.xdata))
        idx = 0
        while idx < len(self.ptrs): # go through each doc sequence sample
            e_idx = idx + self.batch_size
            if e_idx > len(self.ptrs):
                e_idx = len(self.ptrs)
            indices = self.ptrs[idx:e_idx]
            x_mb = []
            max_seq_len = 0
            for i in range(len(indices)):
                ii = indices[i]
                seq_ii = self.xdata[ii]
                max_seq_len = max(max_seq_len, len(seq_ii))
                x_mb.append(seq_ii)
            # do any sequence mini-batch padding to ensure equal length
            for i in range(len(indices)):
                seq_i = x_mb[i]
                if len(seq_i) < max_seq_len:
                    diff = max_seq_len - len(seq_i)
                    for d in range(diff):
                        seq_i.append(-1)
                x_mb[i] = seq_i
            x_mb = np.asarray(x_mb)
            yield x_mb
            idx = e_idx
