from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.examples.tutorials.mnist import  input_data
from tensorflow.python.ops import rnn , rnn_cell
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)


BATCH_SIZE = 128
N_CLASSES = 10

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None, N_CLASSES])





def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def maxpool2d(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')


def convolutional_neural_network(x):
    weights = {
        # 5 x 5 convolution , 1 input image , 32 outputs
        'W_conv1': tf.Variable(tf.random_normal([5,5,1,32])),
        'W_conv2': tf.Variable(tf.random_normal([5,5,32,64])),
        'W_fc'   : tf.Variable(tf.random_normal([7*7*64, 1024])),
        'out'    : tf.Variable(tf.random_normal([1024, N_CLASSES]))
    }

    biases = {
        'b_conv1' : tf.Variable(tf.random_normal([32])),
        'b_conv2': tf.Variable(tf.random_normal([64])),
        'b_fc': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([N_CLASSES]))
    }

    x = tf.reshape(x , shape=[-1,28,28,1])

    conv1 = tf.nn.relu(conv2d(x,weights['W_conv1'])+biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    conv2 = tf.nn.relu(conv2d(conv1,weights['W_conv2'])+biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2, [-1,7*7*64])
    fc = tf.nn.relu(tf.matmul(fc,weights['W_fc'])+biases['b_fc'])

    output = tf.matmul(fc,weights['out']+biases['out'])

    return output


def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / BATCH_SIZE)):
                epoch_x, epoch_y = mnist.train.next_batch(BATCH_SIZE)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
train_neural_network(x)