import tensorflow as tf
import numpy as np
from model import ARGA, ARVGA, Discriminator, Discriminator2
from optimizer import OptimizerAE, OptimizerVAE, OptimizerAE2
import scipy.sparse as sp
import inspect
from scipy.stats import norm

from preprocessing import preprocess_graph, sparse_to_tuple, construct_feed_dict
flags = tf.app.flags
FLAGS = flags.FLAGS

def get_placeholder(adj):
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'real_distribution': tf.placeholder(dtype=tf.float32, shape=[adj.shape[0], FLAGS.hidden2], name='real_distribution'),
        'real_dist_TV': tf.placeholder(dtype=tf.float32, shape=[adj.shape[0]-1, 595], name='real_dist_TV'),
        'fake_dist_for_d2' : tf.placeholder(dtype=tf.float32, name="fake_dist_for_d2")
    }

    return placeholders

def get_model_2(model_str, placeholders, num_features, num_nodes, features_nonzero):
    discriminator2 = Discriminator2()
    d_real_TV = discriminator2.construct(placeholders['real_dist_TV'])
    model = None
    if model_str == 'arga_ae':
        model = ARGA(placeholders, num_features, features_nonzero)

    elif model_str == 'arga_vae':
        model = ARVGA(placeholders, num_features, num_nodes, features_nonzero)

    return d_real_TV, discriminator2, model


def get_model(model_str, placeholders, num_features, num_nodes, features_nonzero):
    discriminator = Discriminator()
    d_real = discriminator.construct(placeholders['real_distribution'])
    model = None
    if model_str == 'arga_ae':
        model = ARGA(placeholders, num_features, features_nonzero)

    elif model_str == 'arga_vae':
        model = ARVGA(placeholders, num_features, num_nodes, features_nonzero)

    return d_real, discriminator, model

def format_data_new(adj, features):
    # Store original adjacency matrix (without diagonal entries) for later
    adj_orig = adj
    adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
    adj_orig.eliminate_zeros()
    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    num_nodes = adj.shape[0]

    features = sparse_to_tuple(features.tocoo())
    num_features = features[2][1]
   
    features_nonzero = features[1].shape[0]
    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
   
    adj_label = adj + sp.eye(adj.shape[0])
    adj_label = sparse_to_tuple(adj_label)
    values = [adj, num_features, num_nodes, features_nonzero, pos_weight, norm, adj_norm, adj_label, features, adj_orig]
    keys = ['adj', 'num_features', 'num_nodes', 'features_nonzero', 'pos_weight', 'norm', 'adj_norm', 'adj_label', 'features', 'adj_orig']
    feas = {}
    feas = dict(zip(keys, values))

    return feas


def get_optimizer_2(model_str, model, discriminator, discriminator2, placeholders, pos_weight, norm, d_real_TV, num_nodes):
    if model_str == 'arga_ae':
        d_fake = discriminator2.construct(placeholders['fake_dist_for_d2'], reuse=True)
        firstZ = discriminator.construct(model.embeddings, reuse=True)
        opt = OptimizerAE2(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm,
                          d_real=d_real_TV,
                          d_fake=d_fake,
                          firstZ=firstZ)
    elif model_str == 'arga_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm,
                           d_real=d_real_TV,
                           d_fake=d_fake)
    return opt

def get_optimizer(model_str, model, discriminator, placeholders, pos_weight, norm, d_real, num_nodes):
    if model_str == 'arga_ae':
        d_fake = discriminator.construct(model.embeddings, reuse=True)
        opt = OptimizerAE(preds=model.reconstructions,
                          labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                      validate_indices=False), [-1]),
                          pos_weight=pos_weight,
                          norm=norm,
                          d_real=d_real,
                          d_fake=d_fake)
    elif model_str == 'arga_vae':
        opt = OptimizerVAE(preds=model.reconstructions,
                           labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                       validate_indices=False), [-1]),
                           model=model, num_nodes=num_nodes,
                           pos_weight=pos_weight,
                           norm=norm,
                           d_real=d_real,
                           d_fake=discriminator.construct(model.embeddings, reuse=True))
    return opt

#emb, avg_cost = update(ae_model, opt, sess, feas['adj_norm'], feas['adj_label'], feas['features'], placeholders, feas['adj'])

def update(model, opt, sess, adj_norm, adj_label, features, placeholders, adj, prior, hiddenSIMLR, new_fake_d):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    feed_dict.update({placeholders['dropout']: 0})
    emb = sess.run(model.z_mean, feed_dict=feed_dict)

    # prior can be (1) features or (2) original_train_TV for the second discriminator
    featureAverage = np.mean(prior, axis=1)
    (mu, sigma) = norm.fit(featureAverage)
    
    if(hiddenSIMLR == "No_hidden_SIMLR"):
        z_real_dist_prior = np.random.normal(mu, sigma, (adj.shape[0], FLAGS.hidden2))
        feed_dict.update({placeholders['real_distribution']: z_real_dist_prior})
    else:
        z_real_dist_prior = np.random.normal(mu, sigma, (prior.shape[0], 595))
        feed_dict.update({placeholders['real_dist_TV']: z_real_dist_prior})
        feed_dict.update({placeholders['fake_dist_for_d2']: new_fake_d})


    for j in range(5):
        _, reconstruct_loss = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
    d_loss, _ = sess.run([opt.dc_loss, opt.discriminator_optimizer], feed_dict=feed_dict)
    g_loss, _ = sess.run([opt.generator_loss, opt.generator_optimizer], feed_dict=feed_dict)

    avg_cost = reconstruct_loss

    return emb, avg_cost


def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]
