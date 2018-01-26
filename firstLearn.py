import tensorflow as tf
import numpy as np

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

var = bias_variable([2,3])
sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())
print(sess.run(var))
# evaluate training accuracy
# curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
# print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
