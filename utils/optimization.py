# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import tensorflow as tf
import time

import utils.plackettluce as pl

def optimize_policy(model, optimizer,
                    data_train, train_doc_weights, train_alpha, train_beta,
                    data_vali,  vali_doc_weights, vali_alpha, vali_beta,
                    n_grad_samples = 'dynamic',
                    n_eval_samples = 100,
                    max_epochs = 50,
                    early_stop_diff = 0.001,
                    early_stop_per_epochs = 3,
                    print_updates=False):
  
  dynamic_n_grad_samples = n_grad_samples == 'dynamic'

  early_stop_per_epochs = min(early_stop_per_epochs, max_epochs)

  stacked_alphas = np.stack([train_alpha, vali_alpha], axis=-1)
  stacked_betas = np.stack([train_beta, vali_beta], axis=-1)

  cutoff = stacked_alphas.shape[0]

  policy_vali_scores = model(data_vali.feature_matrix)[:,0].numpy()
  metrics =  pl.datasplit_metrics(
                  data_vali,
                  policy_vali_scores,
                  stacked_alphas,
                  stacked_betas,
                  vali_doc_weights,
                  n_samples=n_eval_samples,
                )
  if print_updates:
    print('epoch %d: train %0.04f vali %0.04f' % (0, metrics[0], metrics[1]))
  first_metric_value = metrics[1]
  last_metric_value = metrics[1]

  best_metric_value = metrics[1]
  best_weights = model.get_weights()

  cum_doc_weights = np.cumsum(np.abs(train_doc_weights))
  start_weights = cum_doc_weights[data_train.doclist_ranges[:-1]]
  end_weights = cum_doc_weights[data_train.doclist_ranges[1:]-1]
  qid_included = np.where(np.not_equal(start_weights, end_weights))[0]
  qid_included = np.random.permutation(qid_included)

  start_time = time.time()
  n_queries = qid_included.shape[0]
  if dynamic_n_grad_samples:
    n_grad_samples_start = 10.
    n_grad_samples_add_per_step = 90/(10*n_queries)
  for i in range(n_queries*max_epochs):
    qid = qid_included[i%n_queries]

    q_doc_weights = data_train.query_values_from_vector(qid, train_doc_weights)
    q_feat = data_train.query_feat(qid)
    q_cutoff = min(cutoff, data_train.query_size(qid))

    if dynamic_n_grad_samples:
      n_grad_samples = int(np.ceil(n_grad_samples_start + n_grad_samples_add_per_step*i))

    with tf.GradientTape() as tape:
      tf_scores = model(q_feat)[:, 0]
      scores = tf_scores.numpy()

      gradient = pl.gradient_based_on_samples(
                      train_alpha,
                      q_doc_weights,
                      scores,
                      n_samples=n_grad_samples)

      loss = -tf.reduce_sum(tf_scores * gradient)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # reshuffle queries every epoch
    if (i+1) % (n_queries) == 0:
      qid_included = np.random.permutation(qid_included)
    if (i+1) % (n_queries*early_stop_per_epochs) == 0:
      epoch_i = (i+1) / n_queries
      policy_vali_scores = model(data_vali.feature_matrix)[:,0].numpy()
      metrics =  pl.datasplit_metrics(
                      data_vali,
                      policy_vali_scores,
                      stacked_alphas,
                      stacked_betas,
                      vali_doc_weights,
                      n_samples=n_eval_samples,
                    )
      abs_improvement = metrics[1] - last_metric_value
      if print_updates:
        improvement = metrics[1]/last_metric_value - 1.
        total_improvement = metrics[1]/first_metric_value - 1.
        average_time = (time.time() - start_time)/(i+1.)*n_queries
        print('epoch %d: '
              'train %0.04f '
              'vali %0.04f '
              'epoch-time %0.04f '
              'abs-improvement %0.05f '
              'improvement %0.05f '
              'total-improvement %0.05f ' % (epoch_i, 
                 metrics[0], metrics[1],
                 average_time,
                 abs_improvement, improvement, total_improvement)
              )
      last_metric_value = metrics[1]
      if best_metric_value < metrics[1]:
        best_weights = model.get_weights()
      if abs_improvement < early_stop_diff:
        break

  model.set_weights(best_weights)
  return model, last_metric_value   

def optimize_regressor_counterfactual_param_given(
                    model, optimizer,
                    data_train,  train_query_freq, train_clicks, train_exp_alpha, train_exp_beta,
                    data_vali,  vali_query_freq, vali_clicks, vali_exp_alpha,  vali_exp_beta,
                    max_epochs = 500,
                    early_stop_diff = 0.000,
                    early_stop_per_epochs = 3,
                    print_updates=False,
                    alpha_clip=None):
  doc_mask = np.greater(train_exp_alpha, 0)
  n_doc_included = np.sum(doc_mask, dtype=np.int64)

  doc_ids = np.arange(train_clicks.shape[0])[doc_mask]
  doc_clicks = np.sum(train_clicks[doc_mask], axis=1)
  doc_alpha = train_exp_alpha[doc_mask]
  doc_beta = train_exp_beta[doc_mask]

  train_q_d_i = data_train.query_index_per_document()
  doc_query_freq = train_query_freq[train_q_d_i[doc_mask]]
  doc_ctr = doc_clicks/np.maximum(doc_query_freq, 1.)

  vali_doc_mask = np.greater(vali_exp_alpha, 0)
  vali_n_doc_included = np.sum(vali_doc_mask, dtype=np.int64)

  vali_doc_ids = np.arange(vali_clicks.shape[0])[vali_doc_mask]
  vali_doc_clicks = np.sum(vali_clicks[vali_doc_mask], axis=1)
  vali_doc_alpha = vali_exp_alpha[vali_doc_mask]
  vali_doc_beta = vali_exp_beta[vali_doc_mask]

  vali_q_d_i = data_vali.query_index_per_document()
  vali_doc_query_freq = vali_query_freq[vali_q_d_i[vali_doc_mask]]
  vali_doc_ctr = vali_doc_clicks/np.maximum(vali_doc_query_freq, 1.)

  batch_size = 1024

  best_weights = model.get_weights()
  best_loss_value = np.inf

  total_eval_loss = 0.
  for i in range(int(np.ceil(vali_n_doc_included/batch_size))):
    batch_i = np.arange(i*batch_size,(i+1)*batch_size)
    if (i+1)*batch_size > vali_n_doc_included:
      batch_i = batch_i[:vali_n_doc_included - (i+1)*batch_size]
    
    batch_ctr = vali_doc_ctr[batch_i]
    batch_alpha = vali_doc_alpha[batch_i]
    batch_beta = vali_doc_beta[batch_i]

    batch_feat = data_vali.feature_matrix[vali_doc_ids[batch_i], :]

    batch_pred = model(batch_feat)[:, 0].numpy()

    if alpha_clip is None:
      batch_pos_weights = (batch_ctr - batch_beta)/batch_alpha
      batch_neg_weights = 1. - (batch_ctr - batch_beta)/batch_alpha
    else:
      batch_pos_weights = (batch_ctr - batch_beta)/np.maximum(batch_alpha, alpha_clip)
      batch_neg_weights = (batch_alpha - batch_ctr + batch_beta)/np.maximum(batch_alpha, alpha_clip)
    batch_estimate = batch_pos_weights*np.log(batch_pred) + batch_neg_weights*np.log(1. - batch_pred)

    total_eval_loss += -np.sum(batch_estimate)

  best_loss_value = total_eval_loss/float(vali_n_doc_included)

  print('(0) Mean Eval Loss: %s' % best_loss_value)
  
  for epoch_i in range(max_epochs):
    perm_indices = np.random.permutation(n_doc_included)
    losses = []
    for i in range(int(np.ceil(n_doc_included/batch_size))):
      batch_i = perm_indices[i*batch_size:(i+1)*batch_size]
      batch_ctr = doc_ctr[batch_i]
      batch_alpha = doc_alpha[batch_i]
      batch_beta = doc_beta[batch_i]

      batch_feat = data_train.feature_matrix[doc_ids[batch_i], :]

      with tf.GradientTape() as tape:

        batch_pred = model(batch_feat)[:, 0]

        if alpha_clip is None:
          batch_pos_weights = (batch_ctr - batch_beta)/batch_alpha
          batch_neg_weights = 1. - (batch_ctr - batch_beta)/batch_alpha
        else:
          batch_pos_weights = (batch_ctr - batch_beta)/np.maximum(batch_alpha, alpha_clip)
          batch_neg_weights = (batch_alpha - batch_ctr + batch_beta)/np.maximum(batch_alpha, alpha_clip)
        batch_estimate = batch_pos_weights*tf.math.log(batch_pred) + batch_neg_weights*tf.math.log(1. - batch_pred)

        loss = -tf.reduce_sum(batch_estimate)/float(batch_i.shape[0])

        losses.append(loss.numpy())

      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    average_train_loss = np.mean(losses)

    if (epoch_i+1) % early_stop_per_epochs == 0:

      total_eval_loss = 0.
      for i in range(int(np.ceil(vali_n_doc_included/batch_size))):
        batch_i = np.arange(i*batch_size,(i+1)*batch_size)
        if (i+1)*batch_size > vali_n_doc_included:
          batch_i = batch_i[:vali_n_doc_included - (i+1)*batch_size]
        
        batch_ctr = vali_doc_ctr[batch_i]
        batch_alpha = vali_doc_alpha[batch_i]
        batch_beta = vali_doc_beta[batch_i]

        batch_feat = data_vali.feature_matrix[vali_doc_ids[batch_i], :]

        batch_pred = model(batch_feat)[:, 0].numpy()

        if alpha_clip is None:
          batch_pos_weights = (batch_ctr - batch_beta)/batch_alpha
          batch_neg_weights = 1. - (batch_ctr - batch_beta)/batch_alpha
        else:
          batch_pos_weights = (batch_ctr - batch_beta)/np.maximum(batch_alpha, alpha_clip)
          batch_neg_weights = (batch_alpha - batch_ctr + batch_beta)/np.maximum(batch_alpha, alpha_clip)
        batch_estimate = batch_pos_weights*np.log(batch_pred) + batch_neg_weights*np.log(1. - batch_pred)

        total_eval_loss += -np.sum(batch_estimate)

      eval_loss = total_eval_loss/float(vali_n_doc_included)

      print('(%d) Mean Eval Loss: %s' % (epoch_i+1, eval_loss))

      improvement = best_loss_value - eval_loss
      if eval_loss < best_loss_value:
        best_weights = model.get_weights()
        best_loss_value = eval_loss
      if improvement < early_stop_diff:
        break

  print('finished optimizing regression')
  model.set_weights(best_weights)
  return model, average_train_loss, eval_loss


def optimize_regressor_counterfactual_linear(
                    model, optimizer,
                    data_train,  train_query_freq, train_clicks, train_exp_alpha,
                    data_vali,  vali_query_freq, vali_clicks, vali_exp_alpha,
                    max_epochs = 500,
                    early_stop_diff = 0.000,
                    early_stop_per_epochs = 3,
                    print_updates=False,
                    alpha_clip=None):
  doc_mask = np.greater(train_exp_alpha, 0)
  n_doc_included = np.sum(doc_mask, dtype=np.int64)

  doc_ids = np.arange(train_clicks.shape[0])[doc_mask]
  doc_clicks = np.sum(train_clicks[doc_mask], axis=1)
  doc_alpha = train_exp_alpha[doc_mask]

  train_q_d_i = data_train.query_index_per_document()
  doc_query_freq = train_query_freq[train_q_d_i[doc_mask]]
  doc_ctr = doc_clicks/np.maximum(doc_query_freq, 1.)

  vali_doc_mask = np.greater(vali_exp_alpha, 0)
  vali_n_doc_included = np.sum(vali_doc_mask, dtype=np.int64)

  vali_doc_ids = np.arange(vali_clicks.shape[0])[vali_doc_mask]
  vali_doc_clicks = np.sum(vali_clicks[vali_doc_mask], axis=1)
  vali_doc_alpha = vali_exp_alpha[vali_doc_mask]

  vali_q_d_i = data_vali.query_index_per_document()
  vali_doc_query_freq = vali_query_freq[vali_q_d_i[vali_doc_mask]]
  vali_doc_ctr = vali_doc_clicks/np.maximum(vali_doc_query_freq, 1.)

  batch_size = 1024

  best_weights = model.get_weights()
  best_loss_value = np.inf

  epsilon = 10**-5

  total_eval_loss = 0.
  for i in range(int(np.ceil(vali_n_doc_included/batch_size))):
    batch_i = np.arange(i*batch_size,(i+1)*batch_size)
    if (i+1)*batch_size > vali_n_doc_included:
      batch_i = batch_i[:vali_n_doc_included - (i+1)*batch_size]
    
    batch_ctr = vali_doc_ctr[batch_i]
    batch_alpha = vali_doc_alpha[batch_i]

    batch_feat = data_vali.feature_matrix[vali_doc_ids[batch_i], :]

    batch_pred = model(batch_feat)[:, 0].numpy()

    if alpha_clip is None:
      batch_pos_weights = batch_ctr/batch_alpha
      batch_neg_weights = 1. - batch_ctr/batch_alpha
    else:
      batch_pos_weights = batch_ctr/np.maximum(batch_alpha, alpha_clip)
      batch_neg_weights = 1. - batch_ctr/np.maximum(batch_alpha, alpha_clip)
    batch_estimate = (batch_pos_weights*np.log(batch_pred + epsilon)
                      + batch_neg_weights*np.log(1. - batch_pred + epsilon))

    total_eval_loss += -np.sum(batch_estimate)

  best_loss_value = total_eval_loss/float(vali_n_doc_included)

  print('(0) Mean Eval Loss: %s' % best_loss_value)
  
  for epoch_i in range(max_epochs):
    perm_indices = np.random.permutation(n_doc_included)
    losses = []
    for i in range(int(np.ceil(n_doc_included/batch_size))):
      batch_i = perm_indices[i*batch_size:(i+1)*batch_size]
      batch_ctr = doc_ctr[batch_i]
      batch_alpha = doc_alpha[batch_i]

      batch_feat = data_train.feature_matrix[doc_ids[batch_i], :]

      with tf.GradientTape() as tape:

        batch_pred = model(batch_feat)[:, 0]

        if alpha_clip is None:
          batch_pos_weights = batch_ctr/batch_alpha
          batch_neg_weights = 1. - batch_ctr/batch_alpha
        else:
          batch_pos_weights = batch_ctr/np.maximum(batch_alpha, alpha_clip)
          batch_neg_weights = 1. - batch_ctr/np.maximum(batch_alpha, alpha_clip)
        batch_estimate = (batch_pos_weights*tf.math.log(batch_pred + epsilon)
                          + batch_neg_weights*tf.math.log(1. - batch_pred + epsilon))

        loss = -tf.reduce_sum(batch_estimate)/float(batch_i.shape[0])

        losses.append(loss.numpy())

      gradients = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    average_train_loss = np.mean(losses)

    if (epoch_i+1) % early_stop_per_epochs == 0:

      total_eval_loss = 0.
      for i in range(int(np.ceil(vali_n_doc_included/batch_size))):
        batch_i = np.arange(i*batch_size,(i+1)*batch_size)
        if (i+1)*batch_size > vali_n_doc_included:
          batch_i = batch_i[:vali_n_doc_included - (i+1)*batch_size]
        
        batch_ctr = vali_doc_ctr[batch_i]
        batch_alpha = vali_doc_alpha[batch_i]

        batch_feat = data_vali.feature_matrix[vali_doc_ids[batch_i], :]

        batch_pred = model(batch_feat)[:, 0].numpy()

        if alpha_clip is None:
          batch_pos_weights = batch_ctr/batch_alpha
          batch_neg_weights = 1. - batch_ctr/batch_alpha
        else:
          batch_pos_weights = batch_ctr/np.maximum(batch_alpha, alpha_clip)
          batch_neg_weights = 1. - batch_ctr/np.maximum(batch_alpha, alpha_clip)
        batch_estimate = (batch_pos_weights*np.log(batch_pred + epsilon)
                          + batch_neg_weights*np.log(1. - batch_pred + epsilon))

        total_eval_loss += -np.sum(batch_estimate)

      eval_loss = total_eval_loss/float(vali_n_doc_included)

      print('(%d) Mean Eval Loss: %s' % (epoch_i+1, eval_loss))

      improvement = best_loss_value - eval_loss
      if eval_loss < best_loss_value:
        best_weights = model.get_weights()
        best_loss_value = eval_loss
      if improvement < early_stop_diff or np.isnan(improvement):
        break

  print('finished optimizing regression')
  model.set_weights(best_weights)
  return model, average_train_loss, eval_loss
  