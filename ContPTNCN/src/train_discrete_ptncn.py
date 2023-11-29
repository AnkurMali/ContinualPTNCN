import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
"""
Trains a P-TNCN as a discrete token prediction model -- forecasting over sequences
of one-hot encoded integer symbols.
Implementation of the proposed model in Ororbia et al., 2019 IEEE TNNLS.

@author: Ankur Mali

"""
import time
import sys
import pickle
import math

sys.path.insert(0, 'models/')
sys.path.insert(0, 'utils/')

import tensorflow as tf
import numpy as np
from ptncn_2lyr import PTNCN
from data import Vocab
#from data import Vocab, DataLoader, CLS, SEP, MASK, UNK, PAD
from seq_sampler import DataLoader
seed = 1234
tf.random.set_seed(seed=seed)
np.random.seed(seed)

###########################################################################################################
# define helper functions first....
###########################################################################################################

# small helper function for converting integer sequence to string-tokens
def seq_to_tokens(idx_seq, vocab, mb_idx=0):
    tok_seq = ""
    for i in range(0, idx_seq.shape[0]):
        idx = idx_seq[i][mb_idx].numpy()
        tok_seq += vocab.idx2token(idx) + " "
    return tok_seq

# printing function for theta of an NCN model object
def theta_norms_to_str(model, is_header=False):
    str = ""
    theta = model.collect_params()
    for param_name in theta:
        if is_header is False:
            str += "{0}".format(tf.norm(theta[param_name],ord="euclidean"))
        else:
            str += "{0}".format(param_name)
        str += ","
    str = str[:-1]
    return str

def create_fixed_point(data_set, n_rounds=1, n_seq_total=-1):
    """
        Creates a fixed-point sequence sample to be used to properly/stably track loss
    """
    samp_seq_list = []
    n_seq = 0
    if n_seq_total > 0:
        flag = True
        while flag:
            for tr, ip, sg, mk, nxt_snt_flag in data_set:
                samp_seq_list.append( (tr,ip,sg,mk) )
                n_seq += 1
                if n_seq >= n_seq_total:
                    flag = False
                    break
    else:
        debug_n_rounds = -1
        for r in range(n_rounds):
            n_seq = 0
            for tr, ip, sg, mk, nxt_snt_flag in data_set:
                if debug_n_rounds <= 0:
                    samp_seq_list.append( (tr,ip,sg,mk) )
                else:
                    if r < debug_n_rounds:
                        samp_seq_list.append( (tr,ip,sg,mk) )
    return samp_seq_list

def fast_log_loss(probs, y_ind):
    """
        Ororbia-Mali method for calculating categorical NLL via a fast sparse
        indexing approach.
    """
    loss = 0.0
    py = probs.numpy()
    for i in range(0, y_ind.shape[0]):
        ti = y_ind[i][0] # get ith target in sequence
        if ti >= 0: # entry for masked token, which should be non-negative
            py = probs[i,ti]
            if py <= 0.0:
                py = 1e-8
            loss += np.log(py) # all other columns in row i ( != ti) are 0, so do nothing
    return -loss # return negative summed log probs

def eval_model(model, data_set, debug_step_print=False):
    """
         Strictly evaluates the model/architecture on a fixed-point data sample
    """
    cost = 0.0
    acc = 0.0
    mse = 0.0
    num_seq_processed = 0
    N_tok = 0.0

    for x_seq in data_set:
        log_seq_p = 0.0
        mk = tf.cast(tf.greater_equal(x_seq, 0), dtype=tf.float32) # create mask to block off negative indices
        for t in range(x_seq.shape[1]): # step through sequence
            i_t = np.expand_dims(x_seq[:,t],axis=1) # get token indicies at time t
            m_t = tf.expand_dims(mk[:,t],axis=1)

            x_t = tf.squeeze( tf.one_hot(i_t,depth=vocab.size) ) # convert to one-hot encoding for now...
            if i_t.shape[0] == 1:
                x_t = tf.expand_dims(x_t, axis=0)
            x_logits, x_mu = model.forward(x_t, m_t, is_eval=True, beta=beta, alpha=alpha_e)


            if t >= t_prime:
                if use_low_dim_eval is False:
                    log_seq_p += fast_log_loss(x_mu, i_t)
                    x_pred_t = tf.expand_dims(tf.cast(tf.argmax(x_mu,1),dtype=tf.int32),axis=1)
                    comp = tf.cast(tf.equal(x_pred_t, i_t),dtype=tf.float32) * m_t
                    acc += tf.reduce_sum( comp ) # track numerator for accuracy
                else:
                    log_seq_p += -tf.reduce_sum( tf.math.log(x_mu) * x_t )

                if normalize_by_num_seq is False:
                    N_tok += tf.reduce_sum(m_t)
        model.clear_var_history() # clears out statistical "gunk" inside of model
        cost += log_seq_p
        if normalize_by_num_seq is True:
            N_tok += x_seq.shape[0]

        num_seq_processed += x_seq.shape[0]

        N_S = N_tok
        print("\r >> Evaluated on {0} seq, {1} items - cost = {2}".format(num_seq_processed, N_tok, (cost /(N_S)) ), end="")
    print()
    cost = cost / N_tok
    acc = acc / N_tok
    mse = mse / N_tok
    if calc_bpc is True:
        ppl = cost * (1.0 / np.log(2.0))
    else:
        ppl = tf.exp(cost)

    return cost, acc, ppl, mse

def eval_model_timed(model, train_data, dev_data, subtrain_data=None):
    """
         Wrapped evaluation function that includes wall-clock timing.
    """
    start_v = time.process_time()
    if subtrain_data is not None:
        cost_i, acc_i, ppl_i, mse_i = eval_model(model, subtrain_data)
    else:
        cost_i, acc_i, ppl_i, mse_i = eval_model(model, train_data)
    vcost_i, vacc_i, vppl_i, vmse_i = eval_model(model, dev_data)
    end_v = time.process_time()
    eval_time_v = end_v - start_v
    return cost_i, acc_i, vcost_i, vacc_i, eval_time_v, ppl_i, vppl_i, mse_i, vmse_i

###########################################################################################################
# set simulation meta-parameters
###########################################################################################################

train_fname = "../data/ptb_char/trainX.txt"
subtrain_fname = "../data/ptb_char/subX.txt"
dev_fname = "../data/ptb_char/validX.txt"
vocab = "../data/ptb_char/vocab.txt"
out_dir = "/data_reitter/ago109/data/ptb_char/modelA/"
calc_bpc = True
use_low_dim_eval = False
accum_updates = False
normalize_by_num_seq = False # don't mess with this
out_fun = "softmax"

# training meta-parameters
mb = 50 #2
v_mb = 100 #32
eval_iter = 2000 #5000
t_prime = 1 #0

# meta-parameters for model itself --> manhattan distance
model_form = "ptncn"
n_e = 50
opt_type = "nag" #"sgd" # "nag"
init_type = "normal"
alpha = 0.075
momentum = 0.95 # 0.99
update_radius = 1.0
param_radius = -30 #50 #-25.0 #-1.0
w_decay = -0.0001
hid_dim = 1000 #250
wght_sd = 0.05
err_wght_sd = 0.05
beta = 0.1
gamma = 1
act_fun = "tanh"
alpha_e = 0.001 # set according to IEEE paper

load_model = False
model_fname = "model_best.pkl"
eval_only = False

###########################################################################################################
# initialize the program
###########################################################################################################
moment_v = tf.Variable( momentum )
alpha_v  = tf.Variable( alpha )

# prop up the update rule (or "optimizer" in TF lingo)
if opt_type is "nag":
    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=alpha_v,momentum=moment_v,use_nesterov=True)
elif opt_type is "momentum":
    optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=alpha_v,momentum=moment_v,use_nesterov=False)
elif opt_type is "adam":
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=alpha_v)
elif opt_type is "rmsprop":
    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=alpha_v)
else:
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=alpha_v)

print(" > Creating vocab filter...")
vocab = Vocab(vocab)
out_dim = vocab.size

print(" > Vocab.size = ", vocab.size)
train_data = DataLoader(train_fname, mb)
subtrain_data = DataLoader(subtrain_fname, v_mb)
dev_data = DataLoader(dev_fname, v_mb)

if load_model is True or eval_only is True:
    print(" >> Loading pre-trained model: {0}{1}".format(out_dir,model_fname))
    fd = open("{0}{1}".format(out_dir, model_fname), 'rb')
    model = pickle.load( fd )
    fd.close()
else:
    in_dim = -1 # just leave this alone...
    model = PTNCN("ptncn", out_dim, hid_dim, wght_sd=wght_sd, err_wght_sd=err_wght_sd,act_fun=act_fun,out_fun=out_fun,in_dim=in_dim)
print(" Model.Complexity = {0} synapses".format(model.get_complexity()))

cost_i, acc_i, vcost_i, vacc_i, eval_time_v, ppl_i, vppl_i, mse_i, vmse_i = eval_model_timed(model, train_data, dev_data, subtrain_data=subtrain_data)
print(" -1: Tr.L = {0} Tr.Acc = {1} V.L = {2} V.Acc = {3} Tr.MSE = {4} V.MSE = {5} in {6} s".format(cost_i, acc_i, vcost_i, vacc_i, mse_i, vmse_i, eval_time_v))
vcost_im1 = vcost_i

if eval_only is False:

    log = open("{0}{1}".format(out_dir,"perf.txt"),"w")
    log.write("Iter, Loss, PPL, Acc, VLoss, VPPL, VAcc\n")
    log.flush()
    log.write("{0}, {1}, {2}, {3}, {4}, {5}, {6}\n".format(-1, cost_i, ppl_i, acc_i, vcost_i, vppl_i, vacc_i))
    log.flush()

    norm_log = open("{0}{1}".format(out_dir,"norm_log.txt"),"w")
    norm_log.write("{0}\n".format(theta_norms_to_str(model,is_header=True)))
    norm_log.write("{0}\n".format(theta_norms_to_str(model)))
    norm_log.flush()

    for e in range(n_e):
        num_seq_processed = 0
        N_tok = 0.0
        start = time.process_time()
        ########################################################################
        # Training loop
        ########################################################################
        tick = 0
        for x_seq in train_data:
            mk = tf.cast(tf.greater_equal(x_seq, 0), dtype=tf.float32) # create mask to block off negative indices
            delta_accum = []
            for t in range(x_seq.shape[1]): # step through sequence
                i_t = np.expand_dims(x_seq[:,t],axis=1) # get token indicies at time t
                m_t = tf.expand_dims(mk[:,t],axis=1)

                x_t = tf.squeeze( tf.one_hot(i_t,depth=vocab.size) ) # convert to one-hot encoding for now...
                if i_t.shape[0] == 1:
                    x_t = tf.expand_dims(x_t, axis=0)
                x_logits, x_mu = model.forward(x_t, m_t, is_eval=False, beta=beta, alpha=alpha_e)

                if t >= t_prime:
                    delta = model.compute_updates(m_t, gamma=gamma, update_radius=update_radius)
                    N_mb = x_t.shape[0]
                    for p in range(len(delta)):
                        delta[p] = delta[p] * (1.0/(N_mb * 1.0))

                    if accum_updates is False:
                        optimizer.apply_gradients(zip(delta, model.param_var))

                        if w_decay > 0.0:
                            for p in range(len(delta)):
                                delta_var = delta[p]
                                delta[p] = tf.subtract(delta_var,(delta_var * w_decay))

                        if param_radius > 0.0:
                            for p in range(len(model.param_var)):
                                old_var = model.param_var[p]
                                old_var.assign(tf.clip_by_norm(old_var, param_radius, axes=[1]))
                    else:
                        if len(delta_accum) > 0:
                            for p in range(len(delta)):
                                delta_accum[p] = tf.add(delta_accum[p], delta[p])
                        else:
                            for p in range(len(delta)):
                                delta_accum.append(delta[p])

                N_tok += tf.reduce_sum(m_t)

            if accum_updates is True:
                optimizer.apply_gradients(zip(delta_accum, model.param_var))

                if param_radius > 0.0:
                    for p in range(len(model.param_var)):
                        old_var = model.param_var[p]
                        old_var.assign(tf.clip_by_norm(old_var, param_radius, axes=[1]))
            model.clear_var_history() # clears out statistical "gunk" inside of model
            ########################################################################

            num_seq_processed += x_seq.shape[0]
            tick += x_seq.shape[0]
            print("\r  >> Processed {0} seq, {1} tok ".format(num_seq_processed, N_tok), end="")

            if tick >= eval_iter:
                print()
                cost_i, acc_i, vcost_i, vacc_i, eval_time_v, ppl_i, vppl_i, mse_i, vmse_i = eval_model_timed(model, train_data, dev_data, subtrain_data=subtrain_data)
                print(" {0}: Tr.L = {1} Tr.Acc = {2} V.L = {3} V.Acc = {4} Tr.MSE = {5} V.MSE = {6} in {7} s".format(e, cost_i, acc_i, vcost_i, vacc_i, mse_i, vmse_i, eval_time_v))
                log.write("{0}, {1}, {2}, {3}, {4}, {5}, {6}\n".format(e, cost_i, ppl_i, acc_i, vcost_i, vppl_i, vacc_i))
                log.flush()
                norm_log.write("{0}\n".format(theta_norms_to_str(model)))
                norm_log.flush()

                # save model to disk
                fd = open("{0}model{1}.pkl".format(out_dir,e), 'wb')
                pickle.dump(model, fd)
                fd.close()

                if vcost_i <= vcost_im1:
                    fd = open("{0}model_best.pkl".format(out_dir), 'wb')
                    pickle.dump(model, fd)
                    fd.close()
                    vcost_im1 = vcost_i

                tick = 0
        print()
        ########################################################################
        end = time.process_time()
        train_time = end - start
        print("  -> Trained time = {0} s".format(train_time))
        if tick > 0:
            cost_i, acc_i, vcost_i, vacc_i, eval_time_v, ppl_i, vppl_i, mse_i, vmse_i = eval_model_timed(model, train_data, dev_data, subtrain_data=subtrain_data)
            print(" {0}: Tr.L = {1} Tr.Acc = {2} V.L = {3} V.Acc = {4} Tr.MSE = {5} V.MSE = {6} in {7} s".format(e, cost_i, acc_i, vcost_i, vacc_i, mse_i, vmse_i, eval_time_v))
            log.write("{0}, {1}, {2}, {3}, {4}, {5}, {6}\n".format(e, cost_i, acc_i, ppl_i, vcost_i, vacc_i, vppl_i))
            log.flush()
            norm_log.write("{0}\n".format(theta_norms_to_str(model)))
            norm_log.flush()

            # save model to disk
            fd = open("{0}model{1}.pkl".format(out_dir,e), 'wb')
            pickle.dump(model, fd)
            fd.close()

            if vcost_i <= vcost_im1:
                fd = open("{0}model_best.pkl".format(out_dir), 'wb')
                pickle.dump(model, fd)
                fd.close()
                vcost_im1 = vcost_i

            tick = 0

    log.close()
