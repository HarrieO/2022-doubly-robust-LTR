# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import tensorflow as tf

import utils.nnmodel as nn

def EM_estimate_bias(
                    data_train,
                    data_vali,
                    train_clicks,
                    train_displays,
                    vali_clicks,
                    vali_displays,
                     ):

  train_mask = np.greater(np.sum(train_displays, axis=1), 0)
  vali_mask = np.greater(np.sum(vali_displays, axis=1), 0)

  train_clicks = train_clicks[train_mask]
  train_displays = train_displays[train_mask]
  train_feat = data_train.feature_matrix[train_mask]

  vali_clicks = vali_clicks[vali_mask]
  vali_displays = vali_displays[vali_mask]
  vali_feat = data_vali.feature_matrix[vali_mask]

  train_n_docs = train_displays.shape[0]
  vali_n_docs = vali_displays.shape[0]
  cutoff = train_displays.shape[1]

  doc_n_displays = np.sum(train_displays, axis=1)
  doc_display_weight = doc_n_displays/np.sum(doc_n_displays)
  included_doc = np.arange(train_n_docs)[np.greater(doc_n_displays, 0)]
  n_doc_included = included_doc.shape[0]

  alpha = 1./(np.arange(cutoff, dtype=np.float64)+2.)
  beta = 1./(np.arange(cutoff, dtype=np.float64)+2.)
  train_relevance = np.full(train_n_docs, 0.1, dtype=np.float64)
  vali_relevance = np.full(vali_n_docs, 0.1, dtype=np.float64)

  non_nan_alpha = alpha
  non_nan_beta = beta

  train_non_clicks = train_displays - train_clicks
  vali_non_clicks = vali_displays - vali_clicks

  regression_model_params = {'hidden units': [32, 32], 'final activation': True}
  regression_model = nn.init_model(regression_model_params)
  regression_optimizer = tf.keras.optimizers.Adam()
  regression_model.build(input_shape=data_train.feature_matrix.shape)
  regression_init_weights = regression_model.get_weights()
  batch_size = 1024
  log_epsilon = 0.00001

  for i in range(300):

    P_c1_r1 = (alpha + beta)[None,:]*train_relevance[:,None]
    P_c1_r0 = beta[None,:]*(1.-train_relevance[:,None])
    P_c0_r1 = (1. - alpha - beta)[None,:]*train_relevance[:,None]
    P_c0_r0 = (1. - beta[None,:])*(1.-train_relevance[:,None])

    safe_c1_denom = (P_c1_r0 + P_c1_r1)
    safe_c1_denom[np.equal(safe_c1_denom, 0)] = 1.
    safe_c0_denom = (P_c0_r0 + P_c0_r1)
    safe_c0_denom[np.equal(safe_c0_denom, 0)] = 1.
    P_r0_if_c1 = P_c1_r0/safe_c1_denom
    P_r1_if_c1 = P_c1_r1/safe_c1_denom
    P_r0_if_c0 = P_c0_r0/safe_c0_denom
    P_r1_if_c0 = P_c0_r1/safe_c0_denom

    beta = np.sum(train_clicks*P_r0_if_c1, axis=0)
    beta_denom = beta+np.sum(train_non_clicks*P_r0_if_c0, axis=0)
    beta_denom[np.equal(beta_denom, 0)] = 1.
    beta /= beta_denom

    alpha = np.sum(train_clicks*P_r1_if_c1, axis=0)
    alpha_denom = alpha+np.sum(train_non_clicks*P_r1_if_c0, axis=0)
    alpha_denom[np.equal(alpha_denom, 0)] = 1.
    alpha /= alpha_denom

    alpha = alpha - beta

    if not np.all(np.isfinite(alpha)):
      return non_nan_alpha, non_nan_beta
    else:
      non_nan_alpha = alpha
    if not np.all(np.isfinite(beta)):
      return non_nan_alpha, non_nan_beta
    else:
      non_nan_beta = beta

    losses = []
    n_epochs = np.minimum(100, int(3*np.ceil(train_n_docs/512.)))
    for train_i in range(n_epochs):
      batch_i = np.random.choice(n_doc_included,
                               size=batch_size,
                               replace=True)

      batch_feat = train_feat[batch_i, :]

      with tf.GradientTape() as tape:

        batch_pred = regression_model(batch_feat)[:, 0]

        batch_prob_click = tf.math.multiply_no_nan(
                            tf.math.log(tf.math.maximum(log_epsilon,
                              (alpha+beta)[None,:] * batch_pred[:, None]
                              + beta[None,:]*(1.-batch_pred)[:, None],
                              )), train_clicks[batch_i, :])

        batch_prob_non_click = tf.math.multiply_no_nan(
                            tf.math.log(tf.math.maximum(log_epsilon,
                              (1. - alpha-beta)[None,:] * batch_pred[:, None]
                              + (1.-beta)[None,:]*(1.-batch_pred)[:, None],
                              )), train_non_clicks[batch_i, :])

        batch_loss = batch_prob_click + batch_prob_non_click

        loss = -tf.reduce_sum(batch_loss)/float(batch_i.shape[0])
        losses.append(loss.numpy())

      gradients = tape.gradient(loss, regression_model.trainable_variables)
      regression_optimizer.apply_gradients(zip(gradients, regression_model.trainable_variables))

    model_relevance = regression_model(train_feat)[:, 0].numpy()
    train_relevance = model_relevance

  return alpha, beta

def EM_estimate_linear_bias(
                    data_train,
                    data_vali,
                    train_clicks,
                    train_displays,
                    vali_clicks,
                    vali_displays,
                     ):

  train_mask = np.greater(np.sum(train_displays, axis=1), 0)
  vali_mask = np.greater(np.sum(vali_displays, axis=1), 0)

  train_clicks = train_clicks[train_mask]
  train_displays = train_displays[train_mask]
  train_feat = data_train.feature_matrix[train_mask]

  vali_clicks = vali_clicks[vali_mask]
  vali_displays = vali_displays[vali_mask]
  vali_feat = data_vali.feature_matrix[vali_mask]

  train_n_docs = train_displays.shape[0]
  vali_n_docs = vali_displays.shape[0]
  cutoff = train_displays.shape[1]

  doc_n_displays = np.sum(train_displays, axis=1)
  doc_display_weight = doc_n_displays/np.sum(doc_n_displays)
  included_doc = np.arange(train_n_docs)[np.greater(doc_n_displays, 0)]
  n_doc_included = included_doc.shape[0]

  prop = 1./(np.arange(cutoff, dtype=np.float64)+2.)
  train_relevance = np.full(train_n_docs, 0.1, dtype=np.float64)
  vali_relevance = np.full(vali_n_docs, 0.1, dtype=np.float64)

  non_nan_prop = prop

  train_non_clicks = train_displays - train_clicks
  vali_non_clicks = vali_displays - vali_clicks

  regression_model_params = {'hidden units': [32, 32], 'final activation': True}
  regression_model = nn.init_model(regression_model_params)
  regression_optimizer = tf.keras.optimizers.Adam()
  regression_model.build(input_shape=data_train.feature_matrix.shape)
  regression_init_weights = regression_model.get_weights()
  batch_size = 1024
  log_epsilon = 0.00001

  for i in range(30):

    P_c1_r1 = prop[None,:]*train_relevance[:,None]
    P_c0_r1 = (1. - prop)[None,:]*train_relevance[:,None]
    P_c0_r0 = (1.-train_relevance)[:,None]

    safe_c1_denom = (P_c1_r1)
    safe_c1_denom[np.equal(safe_c1_denom, 0)] = 1.
    safe_c0_denom = (P_c0_r0 + P_c0_r1)
    safe_c0_denom[np.equal(safe_c0_denom, 0)] = 1.
    P_r1_if_c1 = P_c1_r1/safe_c1_denom
    P_r1_if_c0 = P_c0_r1/safe_c0_denom

    prop = np.sum(train_clicks*P_r1_if_c1, axis=0)
    prop_denom = prop+np.sum(train_non_clicks*P_r1_if_c0, axis=0)
    prop_denom[np.equal(prop_denom, 0)] = 1.
    prop /= prop_denom

    if not np.all(np.isfinite(prop)):
      return non_nan_prop
    else:
      non_nan_prop = prop

    losses = []
    n_epochs = np.minimum(100, int(3*np.ceil(train_n_docs/512.)))
    for train_i in range(n_epochs):
      batch_i = np.random.choice(n_doc_included,
                               size=batch_size,
                               replace=True)

      batch_feat = train_feat[batch_i, :]

      with tf.GradientTape() as tape:

        batch_pred = regression_model(batch_feat)[:, 0]

        batch_prob_click = tf.math.multiply_no_nan(
                            tf.math.log(tf.math.maximum(log_epsilon,
                              prop[None,:] * batch_pred[:, None],
                              )), train_clicks[batch_i, :])

        batch_prob_non_click = tf.math.multiply_no_nan(
                            tf.math.log(tf.math.maximum(log_epsilon,
                              (1.- prop[None,:] * batch_pred[:, None]),
                              )), train_non_clicks[batch_i, :])

        batch_loss = batch_prob_click + batch_prob_non_click

        loss = -tf.reduce_sum(batch_loss)/float(batch_i.shape[0])

        losses.append(loss.numpy())

      gradients = tape.gradient(loss, regression_model.trainable_variables)

      regression_optimizer.apply_gradients(zip(gradients, regression_model.trainable_variables))

    model_relevance = regression_model(train_feat)[:, 0].numpy()

    train_relevance = model_relevance

  return prop

def deterministic_EM_estimate_linear_bias(
                    data_train,
                    data_vali,
                    train_clicks,
                    train_query_freq,
                    train_rankings,
                    vali_clicks,
                    vali_query_freq,
                    vali_rankings,
                     ):

  train_query_mask = np.greater(train_query_freq, 0)
  vali_query_mask = np.greater(vali_query_freq, 0)

  train_mask = train_query_mask[data_train.query_index_per_document()]
  vali_mask = vali_query_mask[data_vali.query_index_per_document()]

  train_freq = train_query_freq[data_train.query_index_per_document()]
  vali_freq = vali_query_freq[data_vali.query_index_per_document()]

  train_clicks = train_clicks[train_mask]
  train_freq = train_freq[train_mask]
  train_rankings = train_rankings[train_mask]
  train_feat = data_train.feature_matrix[train_mask]

  vali_clicks = vali_clicks[vali_mask]
  vali_freq = vali_freq[vali_mask]
  vali_rankings = vali_rankings[vali_mask]
  vali_feat = data_vali.feature_matrix[vali_mask]

  train_n_docs = train_clicks.shape[0]
  vali_n_docs = vali_clicks.shape[0]
  cutoff = max(data_train.max_query_size(), data_vali.max_query_size())

  prop = 1./(np.arange(cutoff, dtype=np.float64)+2.)
  train_relevance = np.full(train_n_docs, 0.1, dtype=np.float64)
  vali_relevance = np.full(vali_n_docs, 0.1, dtype=np.float64)

  non_nan_prop = prop

  train_non_clicks = train_freq - train_clicks
  vali_non_clicks = vali_freq - vali_clicks

  regression_model_params = {'hidden units': [32, 32], 'final activation': True}
  regression_model = nn.init_model(regression_model_params)
  regression_optimizer = tf.keras.optimizers.Adam()
  regression_model.build(input_shape=data_train.feature_matrix.shape)
  regression_init_weights = regression_model.get_weights()
  batch_size = 1024
  log_epsilon = 0.00001

  n_EM_steps = 300
  for i in range(n_EM_steps):

    train_prop = prop[train_rankings]
    P_c0_r1 = (1. - train_prop)*train_relevance
    P_c0_r0 = (1.-train_relevance)

    safe_c0_denom = (P_c0_r0 + P_c0_r1)
    safe_c0_denom[np.equal(safe_c0_denom, 0)] = 1.
    P_r1_if_c0 = P_c0_r1/safe_c0_denom

    prop[:] = 0.
    np.add.at(prop, train_rankings, train_clicks)
    prop_denom = np.zeros_like(prop)
    prop_denom += prop
    np.add.at(prop_denom, train_rankings, train_non_clicks*P_r1_if_c0)
    prop_denom[np.equal(prop_denom, 0)] = 1.
    prop /= prop_denom

    if not np.all(np.isfinite(prop)):
      return non_nan_prop
    else:
      non_nan_prop = prop

    if i + 1 < n_EM_steps:
      losses = []
      n_epochs = np.minimum(100, int(3*np.ceil(train_n_docs/512.)))
      for train_i in range(n_epochs):
        batch_i = np.random.choice(train_clicks.shape[0],
                                 size=batch_size,
                                 replace=True)

        batch_feat = train_feat[batch_i, :]

        with tf.GradientTape() as tape:

          batch_pred = regression_model(batch_feat)[:, 0]

          batch_prop = prop[train_rankings[batch_i]]
          batch_prob_click = tf.math.multiply_no_nan(
                              tf.math.log(tf.math.maximum(log_epsilon,
                                batch_prop * batch_pred,
                                )), train_clicks[batch_i])

          batch_prob_non_click = tf.math.multiply_no_nan(
                              tf.math.log(tf.math.maximum(log_epsilon,
                                (1.- batch_prop * batch_pred),
                                )), train_non_clicks[batch_i])

          batch_loss = batch_prob_click + batch_prob_non_click

          loss = -tf.reduce_sum(batch_loss)/float(batch_i.shape[0])

          losses.append(loss.numpy())

        gradients = tape.gradient(loss, regression_model.trainable_variables)

        regression_optimizer.apply_gradients(zip(gradients, regression_model.trainable_variables))

      model_relevance = regression_model(train_feat)[:, 0].numpy()

      train_relevance = model_relevance

  return prop