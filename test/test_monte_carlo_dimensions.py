import tensorflow
import tensorflow as tf

# input dimension: (batch_size, 28*28) = (100, 784)
inputs = tf.random_normal((200, 784), 0, 1, dtype=tf.float32)
print("inputs: {}".format(inputs.get_shape()))

act = tf.tile(tf.expand_dims(inputs, 0), [10, 1, 1])
print("act: {}".format(act.get_shape()))

Wm =  tf.random_normal((784,500), 0, 1, dtype=tf.float32)
print("Wm: {}".format(Wm.get_shape()))
Wv = tf.random_normal((784, 500), 0, 1, dtype=tf.float32)
print("Wv: {}".format(Wv.get_shape()))
Bm = tf.random_normal((1, 500), 0, 1, dtype=tf.float32)
print("Bm: {}".format(Bm.get_shape()))
Bv = tf.random_normal((1, 500), 0, 1, dtype=tf.float32)
print("Bv: {}".format(Bv.get_shape()))
epsW = tf.random_normal((10, 784, 500), 0, 1, dtype=tf.float32)
print("epsW: {}".format(epsW.get_shape()))
epsB = tf.random_normal((10, 1, 500), 0, 1, dtype=tf.float32)
print("epsB: {}".format(epsB.get_shape()))

weights = tf.add(tf.multiply(epsW, tf.exp(0.5*Wv)),Wm)
print("weights: {}".format(weights.get_shape()))
biases = tf.add(tf.multiply(epsB, tf.exp(0.5*Bv)), Bm)
print("biases: {}".format(biases.get_shape()))

tmp1 = tf.einsum('mni,mio->mno', act, weights)
print("tmp1: {}".format(tmp1.get_shape()))

pre = tf.add(tf.einsum('mni,mio->mno', act, weights), biases)
print("pre: {}".format(pre.get_shape()))

act = tf.nn.relu(pre)
print("act: {}".format(act.get_shape()))

act = tf.expand_dims(act, 3)
print("act: {}".format(act.get_shape()))

weights = tf.expand_dims(weights, 1)
print("weights: {}".format(weights.get_shape()))

tmp2 = act * weights
print("tmp2: {}".format(tmp2.get_shape()))

pre = tf.add(tf.reduce_sum(act * weights, 2), biases)
print("weights: {}".format(weights.get_shape()))
