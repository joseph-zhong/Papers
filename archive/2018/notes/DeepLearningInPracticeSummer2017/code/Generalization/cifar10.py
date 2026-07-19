import pdb
import argparse
import tensorflow as tf
import numpy as np
import os
import time
import sys

############################## SET UP CONSTANTS ###############################
SMALL_NET = False

parser = argparse.ArgumentParser(
        description='Train a network with a specific activation function.')
parser.add_argument('-i', '--num-iterations', action='store', default=1000, type=int)
parser.add_argument('-b', '--batch-size', action='store', default=128, type=int)
parser.add_argument('-g', '--gpu', type=str, default='2', help='Device number')
parser.add_argument('-l', '--learning-rate', type=float, default=1e-2, help='G.D. learning rate')

args = parser.parse_args()
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
NUM_ITERATIONS = args.num_iterations
BATCH_SIZE = args.batch_size
DATA_PATH = '/home/pillutla/data/cifar10'
DELTA = 0.75
lr = args.learning_rate
# print('learning rate = %e' % lr)


def network(images, num_classes=10, scope='network', train=True):
    slim = tf.contrib.slim
    activation = tf.nn.relu
    with tf.variable_scope(tf.get_variable_scope(), reuse=not train):
        if SMALL_NET:
            with tf.variable_scope(scope, 'LeNet', [images, num_classes]):
                net = activation(slim.conv2d(images, 32, [5, 5], scope='conv1', activation_fn=None))
                net = slim.max_pool2d(net, [2, 2], 2, scope='pool1')
                net = activation(slim.conv2d(net, 64, [5, 5], scope='conv2', activation_fn=None))
                net = slim.max_pool2d(net, [2, 2], 2, scope='pool2')
                net = slim.flatten(net)
                net = activation(slim.fully_connected(net, 1024, scope='fc3', activation_fn=None))
                logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                              scope='fc4')
        else:
            with tf.variable_scope(scope, 'NIN', [images, num_classes]):
                net = activation(slim.conv2d(images, 192, [5, 5], scope='conv1', activation_fn=None))
                net = activation(slim.conv2d(net, 160, [1, 1], scope='conv1_1', activation_fn=None))
                net = activation(slim.conv2d(net, 96, [1, 1], scope='conv1_2', activation_fn=None))
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool1', padding='SAME')
                net = slim.dropout(net, 0.5, scope='dropout1')
                net = activation(slim.conv2d(net, 192, [5, 5], scope='conv2', activation_fn=None))
                net = activation(slim.conv2d(net, 192, [1, 1], scope='conv2_1', activation_fn=None))
                net = activation(slim.conv2d(net, 192, [1, 1], scope='conv2_2', activation_fn=None))
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool2', padding='SAME')
                net = slim.dropout(net, 0.5, scope='dropout2')
                net = activation(slim.conv2d(net, 192, [3, 3], scope='conv3', activation_fn=None))
                net = activation(slim.conv2d(net, 192, [1, 1], scope='conv3_1', activation_fn=None))
                net = slim.conv2d(net, 10, [1, 1], scope='conv3_2', activation_fn=None)
                logits = slim.flatten(tf.nn.avg_pool(net, [1, 8, 8, 1], [1, 1, 1, 1], padding='VALID', name='avg_pool'))
    return logits

def loss(outputs, labels):
    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=outputs))
        loss_summary = tf.summary.scalar('loss', loss)

    return loss, loss_summary

learning_rate = tf.placeholder(tf.float32, shape=[])

def training_step(loss):
    #optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    global_step = tf.train.create_global_step()
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

############################## EVALUATE NETWORK ON TEST SET ####################
def test_net_old(output_op, inputs, labels, iteration, sess):
    num_correct = 0
    num_tested = num_gt_test / BATCH_SIZE * BATCH_SIZE
    for batch in xrange(num_gt_test / BATCH_SIZE):
        batch_input = data_test[
                batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE,
                :, :]
        batch_labels = labels_test[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
        feed_dict = {inputs: batch_input, labels: batch_labels}
        outputs = sess.run(output_op, feed_dict=feed_dict)
        predictions = np.argmax(outputs.squeeze(), axis=1)
        num_correct += np.sum(predictions == batch_labels.squeeze())
    print ('\nIteration %d Accuracy: %d / %d = %0.f%%\n' %
            (iteration, num_correct, num_tested,
                num_correct * 100.0 / num_tested))
    return float(num_correct) / num_tested


def test_net(loss_op, output_op, inputs, labels, iteration, sess):
    num_correct = 0
    num_tested = num_gt_test / BATCH_SIZE * BATCH_SIZE
    for batch in xrange(num_gt_test / BATCH_SIZE):
        batch_input = data_test[
                batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE,
                :, :]
        batch_labels = labels_test[batch * BATCH_SIZE:(batch + 1) * BATCH_SIZE]
        feed_dict = {inputs: batch_input, labels: batch_labels}
        outputs = sess.run(output_op, feed_dict=feed_dict)
        loss = sess.run(loss_op, feed_dict=feed_dict)
        predictions = np.argmax(outputs.squeeze(), axis=1)
        num_correct += np.sum(predictions == batch_labels.squeeze())
    #print ('\nIteration %d Accuracy: %d / %d = %0.f%%\n' %
    #        (iteration, num_correct, num_tested,
    #            num_correct * 100.0 / num_tested))
    return loss, float(num_correct) / num_tested

############################## LOAD TENSORFLOW ################################
config = tf.ConfigProto(log_device_placement=False,
                        allow_soft_placement=True)
config.gpu_options.allow_growth=True

sess = tf.Session(config=config)

############################## CREATE NETWORK #################################

inputs = tf.placeholder(tf.float32, [BATCH_SIZE, 32, 32, 3])
labels = tf.placeholder(tf.int32, [BATCH_SIZE])
accuracy_placeholder = tf.placeholder(tf.float32, [1])
accuracy_summary_op =  tf.summary.scalar('val_accuracy', tf.squeeze(accuracy_placeholder))
outputs = network(inputs)

outputs_test = outputs

loss_op, loss_summary_op = loss(outputs, labels)
training_op = training_step(loss_op)

init = tf.global_variables_initializer()
sess.run(init)
if SMALL_NET:
    log_path = 'logs/lenet/1/%e' % lr
else:
    log_path = 'logs/NIN/1/%e' % lr
summary_writer = tf.summary.FileWriter(log_path, sess.graph)

############################## LOAD TRAIN DATA ################################
import cPickle
data_batches = []
label_batches = []
for batch_id in range(1,6):
    data_batch = cPickle.load(open(os.path.join(DATA_PATH, 'cifar-10-batches-py', 'data_batch_' + str(batch_id))))
    data_batches.append(data_batch['data'])
    label_batches.append(data_batch['labels'])

data_train = np.reshape(np.array(data_batches), (-1, 3, 32, 32)).transpose(0, 2, 3, 1)
train_mean = np.mean(data_train, axis=0)
data_train = (data_train.astype(np.float32) - train_mean) * 2
labels_train = np.array(label_batches).flatten()
# print 'data_train is shape', data_train.shape
# print 'labels_train is shape', labels_train.shape
num_gt_train = data_train.shape[0]
NUM_IT_PER_EPOCH = int( float(num_gt_train) / BATCH_SIZE)

############################## LOAD TEST DATA #################################

data_batch = cPickle.load(open(os.path.join(DATA_PATH, 'cifar-10-batches-py', 'test_batch')))
data_test = np.reshape(data_batch['data'], (-1, 3, 32, 32)).transpose(0, 2, 3, 1)
data_test = (data_test.astype(np.float32) - train_mean) * 2
labels_test = np.array(data_batch['labels'])
# print 'data_test is shape', data_test.shape
# print 'labels_test is shape', labels_test.shape
num_gt_test = data_test.shape[0]

sess.graph.finalize()

############################## RUN SOLVER ITERATIONS ##########################
solverTime = 0
best_acc = 0.0
for iteration in range(NUM_ITERATIONS):
    for inner_iter in range(NUM_IT_PER_EPOCH):
        solverStart = time.time()
        random_inds = np.random.choice(num_gt_train, size=BATCH_SIZE, replace=True)
        batch_input = data_train[random_inds, ...]
        batch_labels = labels_train[random_inds]

        feed_dict = {inputs: batch_input, labels: batch_labels, learning_rate: lr}

        # Run a solver iteration.
        if iteration == 0 and inner_iter == 10:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            _, summary_str = sess.run([training_op, loss_summary_op], feed_dict=feed_dict,
                    options=run_options, run_metadata=run_metadata)
            summary_writer.add_run_metadata(run_metadata, 'step_%07d' % iteration)
        else:
            _, summary_str = sess.run([training_op, loss_summary_op], feed_dict=feed_dict)

        summary_writer.add_summary(summary_str, iteration)

        solverTime += time.time() - solverStart

    if iteration % 10 == 0 or True:
        # Test the solver
        loss, accuracy = test_net(loss_op, outputs_test, inputs, labels, iteration, sess)
        print ('%d %0.3f %0.3f %e' % (iteration, loss, 1 - accuracy, lr))
        if (best_acc > accuracy):
            # performance has not improved, cut lr
            lr = lr * DELTA
            print >> sys.stderr, '\tupdated lr to %e, best_acc = %0.3f, current_acc = %0.3f' % (lr, best_acc, accuracy)
        else:
            best_acc = accuracy

        summary_str = sess.run(accuracy_summary_op, feed_dict={accuracy_placeholder : [accuracy]})
        summary_writer.add_summary(summary_str, iteration)

print >> sys.stderr, '\nDone Solving in %0.3f seconds\n' % solverTime
