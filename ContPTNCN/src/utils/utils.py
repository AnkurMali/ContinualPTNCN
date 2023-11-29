"""
General functional utilities file

@author: Ankur Mali, Alex Ororbia
"""
import tensorflow as tf
import numpy as np

seed = 1234
tf.random.set_seed(seed=seed)
np.random.seed(seed)

def init_weights(init_type, shape, seed, stddev=1):
	if init_type is "glorot_normal":
		initializer = tf.compat.v1.keras.initializers.glorot_normal()
		params = initializer(shape) #, seed=seed )
	elif init_type is "glorot_uniform":
		initializer = tf.compat.v1.keras.initializers.glorot_uniform()
		params = initializer(shape) #, seed=seed )
	elif init_type is "orthogonal":
		initializer = tf.compat.v1.keras.initializers.orthogonal()
		params = initializer(shape)
	elif init_type is "truncated_normal":
		params = tf.random.truncated_normal(shape, stddev=stddev, seed=seed)
	else:
		params = tf.random.normal(shape, stddev=stddev, seed=seed)

	return params

def standardize(x):
	eps=1e-12
	u = tf.reduce_mean(x, keepdims=True)
	s = tf.reduce_mean(tf.pow(x - u, 2), axis=-1, keepdims=True)
	x = (x - u) / tf.sqrt(s + eps)
	return x

def gelu1(x): # quick access function for Gelu
	cdf = 0.5 * (1.0 + tf.tanh(
		(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
	return x * cdf
	#return x * 0.5 * (1.0 + tf.math.erf(x / tf.math.sqrt(2.0)))

def softmax(x):
	"""
		Softmax function with overflow control built in directly
	"""
    max_x = tf.expand_dims( tf.reduce_max(x, axis=1), axis=1)
    exp_x = tf.exp(tf.subtract(x, max_x))
    return exp_x / tf.expand_dims( tf.reduce_sum(exp_x, axis=1), axis=1)

def convert_to_multihot(ind, zero_pad):
	vec = zero_pad.numpy()
	for i in range(0, ind.shape[0]):
		for j in range(0, ind.shape[1]):
			ti = ind[i][j]
			vec[i][ti] = 1.0
	vec = tf.compat.v2.convert_to_tensor(vec, dtype=np.float32)
	return vec

def d_sigmoid(x):
	sigm_x = tf.nn.sigmoid(x)
	return (-sigm_x + 1.0) * sigm_x

def d_tanh(x):
	tanh_x = tf.nn.tanh(x)
	return -(tanh_x * tanh_x) + 1.0

def d_relu(x):
	# df/dx = 1 if 0<x<6 else 0
	val = tf.math.greater(x, 0.0)
	return tf.cast(val,dtype=tf.float32)

def d_relu6(x):
	# df/dx = 1 if 0<x<6 else 0
	# I_x = (z >= a_min) *@ (z <= b_max) //create an indicator function  a = 0 b = 6
	Ix1 = tf.cast(tf.math.greater_equal(x, 0.0),dtype=tf.float32)
	Ix2 = tf.cast(tf.math.less_equal(x, 6.0),dtype=tf.float32)
	Ix = Ix1 * Ix2
	return Ix

def drop_out(input, rate=0.0, seed=69):
	"""
		Custom drop-out function that spits out the binary mask as well
	"""
	mask = tf.math.less_equal( tf.random.uniform(shape=(input.shape[0],input.shape[1]), minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed),(1.0 - rate))
	mask = tf.cast(mask, tf.float32) * (1.0 / (1.0 - rate))
	output = input * mask
	return output, mask

def create_dropout_mask(nrows, ncols, rate=0.0, seed=69):
    mask = tf.math.less_equal( tf.random.uniform(shape=(nrows,ncols), minval=0.0, maxval=1.0, dtype=tf.float32, seed=seed),(1.0 - rate))
    mask = tf.cast(mask, tf.float32) * (1.0 / (1.0 - rate))
    return mask

def clip_by_frobenius(tensor):
	"""
		 Custom Ororbia-Mali-style LRA stability trick function
	"""
	norm2 = tf.norm(tensor, ord='euclidean')
	if norm2 > 1.0:
		tensor = tensor / (norm2 + 1e-6)
	# else do nothing, norm is fine and within the Gaussian ball of radius 1.0
	return tensor

def merge_two_dicts(x, y): #This code is required for python 2.7 upto 3.4
	z = x.copy()   # start with x's keys and values
	z.update(y)    # modifies z with y's keys and values & returns None
	#theta = {**x, **y} For python 3 users
	return z

def sort_input_seq(tok_seq, pos_seq, seg_seq):
	"""
		Performs a complex sort of three lists based on the first list's second element (overrides original lists)
	"""
	tok_seq, pos_seq, seg_seq = zip(*sorted(zip(tok_seq, pos_seq, seg_seq), key=lambda x: x[0][1]))
	return tok_seq, pos_seq, seg_seq

def seq_to_tokens(idx_seq, vocab, mb_idx=0, mask=None):
	tok_seq = ""
	for i in range(0, idx_seq.shape[0]):
		idx = idx_seq[i][mb_idx].numpy()
		if mask is not None:
			if mask[i,mb_idx] > 0:
				tok_seq += vocab.idx2token(idx) + " "
			#else:
			#	tok_seq += "" # "[NIL]" + " "
		else:
			tok_seq += vocab.idx2token(idx) + " "
	return tok_seq

def shuffle_pair(X, Y):
	idx = np.random.permutation(len(X))
	return X[idx], Y[idx]

def ltanh(z):
	a = 1.7159
	b = 2.0/3.0
	z_scale = z * b
	z_scale = tf.clip_by_value(z_scale, -85.0, 85.0)
	neg_exp = tf.exp(-z_scale)
	pos_exp = tf.exp(z_scale)
	denom = tf.add(pos_exp, neg_exp)
	numer = tf.subtract(pos_exp, neg_exp)
	return tf.math.divide(numer, denom) * a

def d_ltanh(z):
	a = 1.7159
	b = 2.0/3.0
	z_scale = z * b
	z_scale = tf.clip_by_value(z_scale, -85.0, 85.0)
	neg_exp = tf.exp(-z_scale)
	pos_exp = tf.exp(z_scale)
	denom = tf.add(pos_exp, neg_exp)
	dx = tf.math.divide((4.0 * a * b), denom * denom)
	return dx

def gte(x):
	return tf.cast(tf.greater_equal(x, 0.0),dtype=tf.float32)
