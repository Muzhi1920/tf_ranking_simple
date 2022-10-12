import lib.losses
import tensorflow as tf

labels = tf.constant([[0.9, 0.8], [0.8, 0.7], [0.5, 0.6], [0.2, 0.1]])

logits = tf.constant([[0.7, 0.5], [0.4, 0.6], [0.3, 0.5], [0.5, 0.3]])


a = tf.expand_dims(labels, 2)
b = tf.expand_dims(labels, 1)
c = tf.subtract(a,b)

print(lib.losses._pairwise_hinge_loss(labels=labels, logits=logits))

