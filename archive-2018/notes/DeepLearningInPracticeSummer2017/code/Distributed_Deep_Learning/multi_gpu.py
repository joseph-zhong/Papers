import tensorflow as tf

def create_clones(num_gpus, model_fn, args=None, kwargs=None):
    inputs = tf.split(args[0], num_gpus, axis=0)
    labels = tf.split(args[1], num_gpus, axis=0)
    args = args or []
    kwargs = kwargs or {}
    # Create clones.
    outputs = []
    # Initial model on CPU
    with tf.name_scope('parameter_server'):
        with tf.device('/cpu:0'):
            output = model_fn(*args, **kwargs)
    for i in range(0, num_gpus):
        args = (inputs[i], labels[i])
        with tf.name_scope('clone_' + str(i)) as clone_scope:
            with tf.device('/gpu:' + str(i)) as clone_device:
                with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                    output = model_fn(*args, **kwargs)
                outputs.append(output)
    return outputs

def average_gradients(tower_grads):
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    grads = []
    for g, _ in grad_and_vars:
      expanded_g = tf.expand_dims(g, 0)
      grads.append(expanded_g)

    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads

def create_multi_gpu_training_op(
        network_fn, network_args, network_kwargs, learning_rate, num_gpus):

    outputs = create_clones(num_gpus, network_fn, network_args, network_kwargs)
    global_step = tf.train.create_global_step()

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    print 'learning_rate', learning_rate

    tower_grads = []
    with tf.variable_scope('tower_grads'):
        for clone_index in xrange(num_gpus):
            grads_and_vars = optimizer.compute_gradients(
                    outputs[clone_index][0], colocate_gradients_with_ops=True)
            tower_grads.append(grads_and_vars)

    with tf.variable_scope('average_grads'):
        with tf.device('/cpu:0'):
            grads = average_gradients(tower_grads)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)

    return train_op, outputs
