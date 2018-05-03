from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.examples.tutorials.mnist import  input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)


n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 64

x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')


def neural_network_model(data):
    hidden_layer_1 = {
        'weights' : tf.Variable(tf.random_normal([784,n_nodes_hl1])),
        'biases'  : tf.Variable(tf.random_normal([n_nodes_hl1]))
    }

    hidden_layer_2 = {
        'weights' : tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
        'biases'  : tf.Variable(tf.random_normal([n_nodes_hl2]))
    }

    hidden_layer_3 = {
        'weights' : tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),
        'biases'  : tf.Variable(tf.random_normal([n_nodes_hl3]))
    }

    output_layer = {
        'weights' : tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),
        'biases'  : tf.Variable(tf.random_normal([n_classes]))
    }

    layer_1 = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
    layer_1 = tf.nn.relu(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, hidden_layer_2['weights']), hidden_layer_2['biases'])
    layer_2 = tf.nn.relu(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, hidden_layer_3['weights']), hidden_layer_3['biases'])
    layer_3 = tf.nn.relu(layer_3)

    output = tf.matmul(layer_3,output_layer['weights'])+output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(mnist.train.num_examples//batch_size):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer,cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch ',epoch, 'completed out of ',hm_epochs,'loss: ',epoch_loss)
        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy: ',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))


train_neural_network(x)
