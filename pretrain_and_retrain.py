import os

import tensorflow as tf
import numpy as np
from random import seed, shuffle

# MLP hyperparameters
N_HIDS = [50, 30, 10]
ACTS = [tf.nn.relu, tf.nn.leaky_relu, tf.nn.relu]

# Dataset hyperparameters
N_ITEMS = 1024
N_DIMS = 30
BATCH_SIZE = 32
N_OUT = 1

# Training hyperparameters
LEARNING_RATE = 1e-3
N_ITERS = 10

# Tensorflow session configuration parameters
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True


def create_two_class_data(n_dims, n_items, m_a=0.0, m_b=2.0, s_a=1.0, s_b=0.5):
    """Create a two-class synthetic dataset of points sampled from Gaussian distributions."""
    inputs = np.concatenate((np.random.normal(loc=m_a, scale=s_a, size=(n_items, n_dims)),
                             np.random.normal(loc=m_b, scale=s_b, size=(n_items, n_dims))))
    labels = np.concatenate((np.ones((n_items, 1)),
                             np.zeros((n_items, 1))))

    np.random.seed(0xbeef)
    np.random.shuffle(inputs)
    np.random.seed(0xbeef)
    np.random.shuffle(labels)

    return (inputs, labels)

def split_into_batches(inputs, labels, batch_size=32):
    """Split data-matrix into a list of batches."""
    return zip([inputs[i*batch_size:(i+1)*batch_size, :]
                for i in xrange(inputs.shape[0] // batch_size)],
               [labels[i*batch_size:(i+1)*batch_size, :]
                for i in xrange(labels.shape[0] // batch_size)])

def initialise_mlp(n_dims, n_hids, acts):
    """Initialise an MLP graph."""
    def mlp(inputs, n_hiddens, activations, reuse=False):
        """Initialise MLP."""

        with tf.variable_scope("mlp_classifier") as mlp_scope:

            layer = tf.layers.dense(
                inputs,
                units=n_hiddens[0],
                activation=activations[0],
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='input_layer')

            for idx, (n_hid, act) in enumerate(zip(n_hiddens[1:], activations[1:])):
                layer = tf.layers.dense(
                    layer,
                    units=n_hid,
                    activation=act,
                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                    name='layer' + str(idx))

            output = tf.layers.dense(
                layer,
                units=1,
                activation=None,
                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                name='output_layer')

        return output

    X = tf.placeholder(name='X', dtype=tf.float32, shape=[None, n_dims])
    Y_gt = tf.placeholder(name='Y_gt', dtype=tf.float32, shape=[None, 1])

    Y_pr = mlp(X, n_hids, acts)

    loss = tf.losses.sigmoid_cross_entropy(Y_gt, Y_pr)
    updates = tf.train.AdamOptimizer().minimize(loss)

    return X, Y_gt, loss, updates

def training_loop(data, session, loss, updates, X, Y_gt, n_iters):
    """Simple training loop."""
    for itr in range(n_iters):
        shuffle(pretrain_data)
        loss_vals = []
        for batch in pretrain_data:
            loss_val, _ = session.run(
                [loss, updates],
                feed_dict={X: batch[0], Y_gt: batch[1]})
            loss_vals.append(loss_val)
        print "Iteration %d, training loss %.3f" % (itr, np.mean(loss_vals))

def pretraining(data, n_dims, n_hids, acts, n_iters, save_path='pretrain_MLP'):
    """Randomly initialise and train a model."""
    # Create a new graph for the model in its pretraining stage.
    pretrain_graph = tf.Graph()
    with pretrain_graph.as_default():
        X, Y_gt, loss, updates = initialise_mlp(n_dims, n_hids, acts)

        with tf.Session(config=config) as pretrain_session:
            pretrain_session.run(tf.global_variables_initializer())

            training_loop(data, pretrain_session, loss, updates, X, Y_gt, n_iters)

            pretrain_saver = tf.train.Saver()
            pretrain_saver.save(pretrain_session, os.path.join(save_path, 'model'))

    return save_path

def retraining(data, n_dims, n_hids, acts, n_iters, save_path='retrain_MLP'):
    """Initialise a model's parameters to a pre-trained model and train it ."""
    # Load pretrained model and retrieve its parameters as numpy tensors
    loaded_graph = tf.Graph()
    with loaded_graph.as_default():
        with tf.Session(config=config) as load_session:
            saver = tf.train.import_meta_graph(os.path.join(model_path, 'model.meta'))
            saver.restore(load_session, tf.train.latest_checkpoint(model_path))
            loaded_variables = sorted(tf.trainable_variables(), key=lambda v: v.name)

            params_pretrain = dict((var.name, var.eval(session=load_session))
                                     for var in loaded_variables)

    # Create re-train graph and apply the pre-trained model parameters. 
    retrain_graph = tf.Graph()
    with retrain_graph.as_default():
        X, Y_gt, loss, updates = initialise_mlp(n_dims, n_hids, acts)

        assignments = dict((var.name, tf.assign(var, params_pretrain[var.name]))
                           for var in sorted(tf.trainable_variables(), key=lambda v: v.name))

        with tf.Session(config=config) as retrain_session:
            # Initialise all graph variables to random/initialisation values.
            retrain_session.run(tf.global_variables_initializer())

            # IMPORTANT: Update trainable variable values to pre-trained values.
            retrain_session.run(assignments.values())

            training_loop(data, retrain_session, loss, updates, X, Y_gt, n_iters)

            retrain_saver = tf.train.Saver()
            retrain_saver.save(retrain_session, os.path.join(save_path, 'model'))


if __name__ == "__main__":

    tf.set_random_seed(31386)

    # Pretraining step
    pretrain_data = split_into_batches(
            *create_two_class_data(N_DIMS, N_ITEMS, m_a=0.0, m_b=2.0, s_a=2.0, s_b=2.0),
            batch_size=32)
    model_path = pretraining(pretrain_data, N_DIMS, N_HIDS, ACTS, N_ITERS)

    # NOTE: All we have at this stage is the string model_path giving us the path to the saved model

    # Retraining step
    retrain_data = split_into_batches(
            *create_two_class_data(N_DIMS, N_ITEMS, m_a=2.0, m_b=0.0, s_a=0.5, s_b=1.0),
            batch_size=32)
    model_path = retraining(retrain_data, N_DIMS, N_HIDS, ACTS, N_ITERS)
