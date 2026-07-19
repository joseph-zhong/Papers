import pdb
import argparse
import tensorflow as tf
import numpy as np
import os
import time
import multi_gpu

############################## SET UP CONSTANTS ###############################
slim = tf.contrib.slim

parser = argparse.ArgumentParser(
        description='Train a network on multiple gpus.')
parser.add_argument('-i', '--num-iterations', action='store', default=10000, type=int)
parser.add_argument('-b', '--batch-size', action='store', default=256, type=int)
parser.add_argument('-g', '--gpu', type=str, default='0', help='Device number')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
NUM_ITERATIONS = args.num_iterations
num_gpus = len(args.gpu.split(','))
BATCH_SIZE = args.batch_size * num_gpus

def network(images, labels, num_classes=10, scope='network', train=True):
    with tf.variable_scope(tf.get_variable_scope(), reuse=not train):
        with tf.variable_scope(scope, 'NIN', [images, num_classes]):
            with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.elu,
                    weights_regularizer=None):
                net = slim.conv2d(images, 192, [5, 5], scope='conv1')
                net = slim.conv2d(net, 160, [1, 1], scope='conv1_1')
                net = slim.conv2d(net, 96, [1, 1], scope='conv1_2')
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool1', padding='SAME')
                net = slim.dropout(net, 0.5, scope='dropout1', is_training=train)
                net = slim.conv2d(net, 192, [5, 5], scope='conv2')
                net = slim.conv2d(net, 192, [1, 1], scope='conv2_1')
                net = slim.conv2d(net, 192, [1, 1], scope='conv2_2')
                net = slim.max_pool2d(net, [3, 3], 2, scope='pool2', padding='SAME')
                net = slim.dropout(net, 0.5, scope='dropout2', is_training=train)
                net = slim.conv2d(net, 192, [3, 3], scope='conv3')
                net = slim.conv2d(net, 192, [1, 1], scope='conv3_1')
                net = slim.conv2d(net, 10, [1, 1], scope='conv3_2', activation_fn=None)
                logits = slim.flatten(tf.nn.avg_pool(net, [1, 8, 8, 1], [1, 1, 1, 1],
                    padding='VALID', name='avg_pool'))

    with tf.variable_scope('loss'):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits))
        total_loss = loss
        #tf.losses.add_loss(loss)
        #total_loss = tf.losses.get_total_loss()

    return total_loss, loss, logits

def training(loss):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = slim.learning.create_train_op(loss, optimizer)
    return train_op

############################## EVALUATE NETWORK ON TEST SET ####################
def test_net(output_op, inputs, labels, iteration, sess):
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

#total_loss, loss_val, logits = network(inputs, labels)
#training_op = training(total_loss)

training_op, output_ops = multi_gpu.create_multi_gpu_training_op(network, (inputs, labels), {}, 1e-4 * num_gpus, num_gpus=num_gpus)
loss_val = tf.reduce_mean([oo[1] for oo in output_ops])

loss_summary_op = tf.summary.scalar('loss', loss_val)

with tf.name_scope('test_network'):
    outputs_test = network(inputs, labels, train=False)[2]

init = tf.global_variables_initializer()
sess.run(init)
summary_writer = tf.summary.FileWriter('logs/num_gpus_%d' % num_gpus, sess.graph)

############################## LOAD DATA ######################################
if not os.path.exists('labels_train.npy'):
    import cPickle
    data_batches = []
    label_batches = []
    for batch_id in range(1,6):
        data_batch = cPickle.load(open('cifar-10-batches-py/data_batch_' + str(batch_id)))
        data_batches.append(data_batch['data'])
        label_batches.append(data_batch['labels'])

    data_train = np.reshape(np.array(data_batches), (-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    print data_train.min(), data_train.max(), data_train.mean()
    train_mean = np.mean(data_train, axis=0)
    data_train = (data_train.astype(np.float32) - train_mean) / 128.0
    print data_train.min(), data_train.max(), data_train.mean()
    labels_train = np.array(label_batches).flatten()
    print 'data_train is shape', data_train.shape
    print 'labels_train is shape', labels_train.shape
    np.save('data_train.npy', data_train)
    np.save('labels_train.npy', labels_train)

    data_batch = cPickle.load(open('cifar-10-batches-py/test_batch'))
    data_test = np.reshape(data_batch['data'], (-1, 3, 32, 32)).transpose(0, 2, 3, 1)
    data_test = (data_test.astype(np.float32) - train_mean) / 128.0
    labels_test = np.array(data_batch['labels'])
    print 'labels_test is shape', labels_test.shape
    print 'data_test is shape', data_test.shape

    np.save('labels_test.npy', labels_test)
    np.save('data_test.npy', data_test)
else:
    data_train = np.load('data_train.npy')
    labels_train = np.load('labels_train.npy')
    data_test = np.load('data_test.npy')
    labels_test = np.load('labels_test.npy')

num_gt_train = data_train.shape[0]
num_gt_test = data_test.shape[0]

sess.graph.finalize()

############################## RUN SOLVER ITERATIONS ##########################
solverTime = 0
for iteration in range(NUM_ITERATIONS):
    solverStart = time.time()
    random_inds = np.random.choice(num_gt_train, size=BATCH_SIZE, replace=True)
    batch_input = data_train[random_inds, ...]
    batch_labels = labels_train[random_inds]

    feed_dict = {inputs: batch_input, labels: batch_labels}

    # Run a solver iteration.
    if iteration == 10:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        _, summary_str = sess.run([training_op, loss_summary_op], feed_dict=feed_dict,
                options=run_options, run_metadata=run_metadata)
        summary_writer.add_run_metadata(run_metadata, 'step_%07d' % iteration)
    else:
        _, summary_str = sess.run([training_op, loss_summary_op], feed_dict=feed_dict)

    summary_writer.add_summary(summary_str, iteration)

    solverTime += time.time() - solverStart
    if iteration % 100 == 0:
        # Test the solver
        accuracy = test_net(outputs_test, inputs, labels, iteration, sess)
        summary_str = sess.run(accuracy_summary_op, feed_dict={accuracy_placeholder : [accuracy]})
        summary_writer.add_summary(summary_str, iteration)

print '\nDone Solving in %0.3f seconds\n' % solverTime
