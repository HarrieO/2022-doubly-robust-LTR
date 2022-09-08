# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import tensorflow as tf
import time
import utils.ranking as rnk
import utils.plackettluce as pl

def pairwise_weights_from_info(data_split, prop, prop_clip, pairwise_info):

  (pair_ranges,
    clicked_id,
    not_clicked_id,
    clicked_rank,
    not_clicked_rank,
    selected_counter) = pairwise_info

  n_queries = data_split.num_queries()

  new_ranges = np.zeros_like(pair_ranges)

  ratio_prop = np.minimum(prop_clip, prop[not_clicked_rank]/np.maximum(prop[clicked_rank], 10**-6))*selected_counter
  qid_included = np.where(np.not_equal(pair_ranges[:-1],pair_ranges[1:]))[0]

  cur_qid = np.where(np.equal(0, pair_ranges))[0][-1]
  next_q_i = pair_ranges[cur_qid+1]
  store_i = 0
  for i in range(1, pair_ranges[-1]):
    if (clicked_id[i] == clicked_id[store_i]
        and not_clicked_id[i] == not_clicked_id[store_i]
        and i < next_q_i):
      ratio_prop[store_i] += ratio_prop[i]
    else:
      store_i += 1
      clicked_id[store_i] = clicked_id[i]
      not_clicked_id[store_i] = not_clicked_id[i]
      ratio_prop[store_i] = ratio_prop[i]

      if i == next_q_i:
        new_ranges[cur_qid+1] = store_i
        # if cur_qid < n_queries - 1:
        if next_q_i != pair_ranges[cur_qid+2]:
          cur_qid += 1
        else:
          next_qid = cur_qid + 1 + np.where(np.equal(next_q_i, pair_ranges[cur_qid+1:]))[0][-1]
          new_ranges[cur_qid+2:next_qid+1] = store_i
          cur_qid = next_qid
        next_q_i = pair_ranges[cur_qid+1]

  new_ranges[cur_qid+1:] = store_i+1

  clicked_id = clicked_id[:store_i+1]
  not_clicked_id = not_clicked_id[:store_i+1]
  ratio_prop = ratio_prop[:store_i+1]

  for qid in qid_included:
    s_i, e_i = pair_ranges[qid:qid+2]
    new_s_i, new_e_i = new_ranges[qid:qid+2]
    ratio_prop[new_s_i:new_e_i] /= np.sum(selected_counter[s_i:e_i])

  return (new_ranges,
          clicked_id,
          not_clicked_id,
          ratio_prop)

def optimize_policy(model, optimizer, prop,
                    data_train, train_pairwise_info,
                    data_vali, vali_pairwise_info,
                    max_epochs = 50,
                    early_stop_diff = 0.001,
                    early_stop_per_epochs = 3,
                    print_updates=False):

  early_stop_per_epochs = min(early_stop_per_epochs, max_epochs)

  cutoff = prop.shape[0]

  (train_pair_ranges,
    train_clicked_id,
    train_not_clicked_id,
    train_pair_weights) = train_pairwise_info

  (vali_pair_ranges,
    vali_clicked_id,
    vali_not_clicked_id,
    vali_pair_weights) = vali_pairwise_info

  train_qid_included = np.where(np.not_equal(train_pair_ranges[:-1],train_pair_ranges[1:]))[0]
  vali_qid_included = np.where(np.not_equal(vali_pair_ranges[:-1],vali_pair_ranges[1:]))[0]

  vali_loss = 0.
  vali_scores = -model(data_vali.feature_matrix)[:,0].numpy()
  vali_inverted_rankings = rnk.data_split_rank_and_invert(vali_scores, data_vali)[1]
  for qid in vali_qid_included:
    q_scores = data_vali.query_values_from_vector(qid, vali_scores)
    q_inv_ranking = data_vali.query_values_from_vector(qid, vali_inverted_rankings)

    s_i, e_i = vali_pair_ranges[qid:qid+2]
    q_pair_clicked = vali_clicked_id[s_i:e_i]
    q_pair_not_clicked = vali_not_clicked_id[s_i:e_i]
    q_pair_weight = vali_pair_weights[s_i:e_i]

    dcg_weights = -np.abs(np.log2(q_inv_ranking[q_pair_clicked]+2.)
                        - np.log2(q_inv_ranking[q_pair_not_clicked]+2.))
    denom_weights = 1. + np.exp(q_scores[q_pair_clicked] - q_scores[q_pair_not_clicked])

    q_loss = np.sum(dcg_weights/denom_weights*q_pair_weight)
    vali_loss += q_loss 
  
  vali_loss = vali_loss/vali_qid_included.shape[0]

  if print_updates:
    print('epoch %d: vali loss %0.04f' % (0, vali_loss))
  first_metric_value = vali_loss
  last_metric_value = vali_loss

  best_metric_value = vali_loss
  best_weights = model.get_weights()

  start_time = time.time()
  n_queries = train_qid_included.shape[0]
  
  train_qid_included = np.random.permutation(train_qid_included)
  for i in range(n_queries*max_epochs):
    qid = train_qid_included[i%n_queries]

    s_i, e_i = train_pair_ranges[qid:qid+2]
    q_pair_clicked = train_clicked_id[s_i:e_i]
    q_pair_not_clicked = train_not_clicked_id[s_i:e_i]
    q_pair_weight = train_pair_weights[s_i:e_i]
    
    q_feat = data_train.query_feat(qid)

    with tf.GradientTape() as tape:
      q_scores = model(q_feat)[:, 0]
      q_inv_ranking = rnk.rank_and_invert(q_scores.numpy())[1]

      dcg_weights = np.abs((np.log2(q_inv_ranking[q_pair_clicked]+2.)
                          - np.log2(q_inv_ranking[q_pair_not_clicked]+2.)))

      q_clicked_scores = tf.gather(q_scores, q_pair_clicked)
      q_not_clicked_scores = tf.gather(q_scores, q_pair_not_clicked)
      denom_weights = 1. + tf.exp(q_clicked_scores - q_not_clicked_scores)

      loss = tf.reduce_sum(dcg_weights/denom_weights*q_pair_weight)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # reshuffle queries every epoch
    if (i+1) % (n_queries) == 0:
      train_qid_included = np.random.permutation(train_qid_included)
    if (i+1) % (n_queries*early_stop_per_epochs) == 0:
      epoch_i = (i+1) / n_queries

      vali_loss = 0.
      vali_scores = -model(data_vali.feature_matrix)[:,0].numpy()
      vali_inverted_rankings = rnk.data_split_rank_and_invert(vali_scores, data_vali)[1]
      for qid in vali_qid_included:
        q_scores = data_vali.query_values_from_vector(qid, vali_scores)
        q_inv_ranking = data_vali.query_values_from_vector(qid, vali_inverted_rankings)

        s_i, e_i = vali_pair_ranges[qid:qid+2]
        q_pair_clicked = vali_clicked_id[s_i:e_i]
        q_pair_not_clicked = vali_not_clicked_id[s_i:e_i]
        q_pair_weight = vali_pair_weights[s_i:e_i]

        dcg_weights = -np.abs(np.log2(q_inv_ranking[q_pair_clicked]+2.)
                            - np.log2(q_inv_ranking[q_pair_not_clicked]+2.))
        denom_weights = 1. + np.exp(q_scores[q_pair_clicked] - q_scores[q_pair_not_clicked])

        q_loss = np.sum(dcg_weights/denom_weights*q_pair_weight)
        vali_loss += q_loss 
      
      vali_loss = vali_loss/vali_qid_included.shape[0]


      abs_improvement = -(vali_loss - last_metric_value)
      if print_updates:
        improvement = vali_loss/last_metric_value - 1.
        total_improvement = vali_loss/first_metric_value - 1.
        average_time = (time.time() - start_time)/(i+1.)*n_queries
        print('epoch %d: '
              'loss %0.04f '
              'epoch-time %0.04f '
              'abs-improvement %0.05f '
              'improvement %0.05f '
              'total-improvement %0.05f ' % (epoch_i, 
                 vali_loss,
                 average_time,
                 abs_improvement, improvement, total_improvement)
              )
      last_metric_value = vali_loss
      if best_metric_value > vali_loss:
        best_weights = model.get_weights()
      if abs_improvement < early_stop_diff:
        break

  model.set_weights(best_weights)
  return model, last_metric_value

def deterministic_optimize_policy(model, optimizer,
                    data_train,
                    train_clicks,
                    train_query_freq,
                    train_prop,
                    data_vali,
                    vali_clicks,
                    vali_query_freq,
                    vali_prop,
                    alpha_clip,
                    max_epochs = 20,
                    early_stop_diff = 0.001,
                    early_stop_per_epochs = 3,
                    print_updates=False
                    ):
  early_stop_per_epochs = min(early_stop_per_epochs, max_epochs)


  n_train_queries = data_train.num_queries()
  n_vali_queries = data_vali.num_queries()

  train_squared_indices = np.zeros(n_train_queries+1, dtype=np.int64)
  train_squared_indices[1:] = np.cumsum(data_train.query_sizes()**2)
  vali_squared_indices = np.zeros(n_vali_queries+1, dtype=np.int64)
  vali_squared_indices[1:] = np.cumsum(data_vali.query_sizes()**2)


  train_qid_included = np.greater(train_query_freq, 0)
  vali_qid_included = np.greater(vali_query_freq, 0)
  for qid in range(n_train_queries):
    if train_qid_included[qid]:
      s_i, e_i = train_squared_indices[qid:qid+2]
      train_qid_included[qid] = np.any(train_clicks[s_i:e_i])
  for qid in range(n_vali_queries):
    if vali_qid_included[qid]:
      s_i, e_i = vali_squared_indices[qid:qid+2]
      vali_qid_included[qid] = np.any(vali_clicks[s_i:e_i])
  train_qid_included = np.where(train_qid_included)[0]
  vali_qid_included = np.where(vali_qid_included)[0]

  for qid in train_qid_included:
    q_n_docs = data_train.query_size(qid)
    q_prop = data_train.query_values_from_vector(qid, train_prop)
    s_i, e_i = train_squared_indices[qid:qid+2]
    q_pairwise = np.reshape(train_clicks[s_i:e_i], (q_n_docs, q_n_docs))
    q_pairwise *= np.minimum(q_prop[None,:]/np.maximum(q_prop[:,None],10.**-5.), 1./alpha_clip)
    train_clicks[s_i:e_i] = q_pairwise.flatten()

  for qid in vali_qid_included:
    q_n_docs = data_vali.query_size(qid)
    q_prop = data_vali.query_values_from_vector(qid, vali_prop)
    s_i, e_i = vali_squared_indices[qid:qid+2]
    q_pairwise = np.reshape(vali_clicks[s_i:e_i], (q_n_docs, q_n_docs))
    q_pairwise *= np.minimum(q_prop[None,:]/np.maximum(q_prop[:,None],10.**-5.), 1./alpha_clip)
    vali_clicks[s_i:e_i] = q_pairwise.flatten()

  vali_loss = 0.
  vali_scores = -model(data_vali.feature_matrix)[:,0].numpy()
  vali_inverted_rankings = rnk.data_split_rank_and_invert(vali_scores, data_vali)[1]
  for qid in vali_qid_included:
    q_n_docs = data_vali.query_size(qid)
    q_scores = data_vali.query_values_from_vector(qid, vali_scores)
    q_inv_ranking = data_vali.query_values_from_vector(qid, vali_inverted_rankings)

    s_i, e_i = vali_squared_indices[qid:qid+2]
    q_pairwise = np.reshape(vali_clicks[s_i:e_i], (q_n_docs, q_n_docs))

    dcg_weights = -np.abs(1./np.log2(q_inv_ranking[:,None]+2.)
                        - 1./np.log2(q_inv_ranking[None,:]+2.))
    denom_weights = 1. + np.exp(q_scores[:,None] - q_scores[None,:])
    q_loss = np.sum(dcg_weights/denom_weights*q_pairwise)
    vali_loss += q_loss 
  
  vali_loss = vali_loss/vali_qid_included.shape[0]

  if print_updates:
    print('epoch %d: vali loss %0.04f' % (0, vali_loss))
  first_metric_value = vali_loss
  last_metric_value = vali_loss

  best_metric_value = vali_loss
  best_weights = model.get_weights()

  start_time = time.time()
  n_queries = train_qid_included.shape[0]
  
  train_qid_included = np.random.permutation(train_qid_included)
  for i in range(n_queries*max_epochs):
    qid = train_qid_included[i%n_queries]

    q_n_docs = data_train.query_size(qid)
    s_i, e_i = train_squared_indices[qid:qid+2]
    q_pairwise = np.reshape(train_clicks[s_i:e_i], (q_n_docs, q_n_docs))
    i1_pairs, i2_pairs = np.where(q_pairwise)
    
    q_feat = data_train.query_feat(qid)

    # with tf.GradientTape() as tape:
    #   q_scores = model(q_feat)[:, 0]
    #   q_inv_ranking = rnk.rank_and_invert(q_scores.numpy())[1]

    #   dcg_weights = np.abs((1./np.log2(q_inv_ranking[:,None]+2.)
    #                       - 1./np.log2(q_inv_ranking[None,:]+2.)))
    #   denom_weights = 1. + tf.exp(q_scores[:,None] - q_scores[None,:])

    #   loss = tf.reduce_sum(dcg_weights/denom_weights*q_pairwise) 

    with tf.GradientTape() as tape:
      q_scores = model(q_feat)[:, 0]
      q_inv_ranking = rnk.rank_and_invert(q_scores.numpy())[1]

      dcg_weights = np.abs((1./np.log2(q_inv_ranking[i1_pairs]+2.)
                          - 1./np.log2(q_inv_ranking[i2_pairs]+2.)))
      denom_weights = 1. + tf.exp(tf.gather(q_scores,i1_pairs) - tf.gather(q_scores,i2_pairs))

      loss = tf.reduce_sum(dcg_weights/denom_weights*q_pairwise[i1_pairs,i2_pairs])

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # reshuffle queries every epoch
    if (i+1.) % (n_queries) == 0:
      train_qid_included = np.random.permutation(train_qid_included)
    if (i+1.) % (n_queries*early_stop_per_epochs) == 0:
      epoch_i = (i+1) / n_queries
      vali_loss = 0.
      vali_scores = -model(data_vali.feature_matrix)[:,0].numpy()
      vali_inverted_rankings = rnk.data_split_rank_and_invert(vali_scores, data_vali)[1]
      for qid in vali_qid_included:
        q_n_docs = data_vali.query_size(qid)
        q_scores = data_vali.query_values_from_vector(qid, vali_scores)
        q_inv_ranking = data_vali.query_values_from_vector(qid, vali_inverted_rankings)

        s_i, e_i = vali_squared_indices[qid:qid+2]
        q_pairwise = np.reshape(vali_clicks[s_i:e_i], (q_n_docs, q_n_docs))

        dcg_weights = -np.abs(1./np.log2(q_inv_ranking[:,None]+2.)
                            - 1./np.log2(q_inv_ranking[None,:]+2.))
        denom_weights = 1. + np.exp(q_scores[:,None] - q_scores[None,:])
        q_loss = np.sum(dcg_weights/denom_weights*q_pairwise)
        vali_loss += q_loss
      
      vali_loss = vali_loss/vali_qid_included.shape[0]


      abs_improvement = -(vali_loss - last_metric_value)
      if print_updates:
        improvement = vali_loss/last_metric_value - 1.
        total_improvement = vali_loss/first_metric_value - 1.
        average_time = (time.time() - start_time)/(i+1.)*n_queries
        print('epoch %d: '
              'loss %0.04f '
              'epoch-time %0.04f '
              'abs-improvement %0.05f '
              'improvement %0.05f '
              'total-improvement %0.05f ' % (epoch_i, 
                 vali_loss,
                 average_time,
                 abs_improvement, improvement, total_improvement)
              )
      last_metric_value = vali_loss
      if best_metric_value > vali_loss:
        best_weights = model.get_weights()
      if abs_improvement < early_stop_diff or np.isnan(abs_improvement):
        break

  model.set_weights(best_weights)
  return model, last_metric_value

def simulate_on_dataset(data_train,
                        data_validation,
                        n_samples,
                        train_doc_weights,
                        validation_doc_weights,
                        alpha,
                        beta,
                        model=None,
                        train_policy_scores=None,
                        vali_policy_scores=None,
                        return_display=False,
                        store_per_rank=False,
                        deterministic_ranking=False
                        ):
  n_train_queries = data_train.num_queries()
  n_vali_queries = data_validation.num_queries()

  train_ratio = n_train_queries/float(n_train_queries + n_vali_queries)

  sampled_queries = np.random.choice(2, size=n_samples,
                                     p=[train_ratio, 1.-train_ratio])
  samples_per_split = np.zeros(2, dtype=np.int64)
  np.add.at(samples_per_split, sampled_queries, 1)

  (train_clicks,
   train_displays,
   train_samples_per_query,
   train_pairwise_info) = simulate_queries(
                     data_train,
                     samples_per_split[0],
                     train_doc_weights,
                     alpha,
                     beta,
                     model=model,
                     all_policy_scores=train_policy_scores,
                     return_display=return_display,
                     store_per_rank=store_per_rank,
                     deterministic_ranking=deterministic_ranking)

  (validation_clicks,
   validation_displays,
   validation_samples_per_query,
   validation_pairwise_info) = simulate_queries(
                     data_validation,
                     samples_per_split[0],
                     validation_doc_weights,
                     alpha,
                     beta,
                     model=model,
                     all_policy_scores=vali_policy_scores,
                     return_display=return_display,
                     store_per_rank=store_per_rank,
                     deterministic_ranking=deterministic_ranking)

  return (train_clicks, train_displays,
          train_samples_per_query, train_pairwise_info,
          validation_clicks, validation_displays,
          validation_samples_per_query, validation_pairwise_info)


def simulate_queries(data_split,
                     n_samples,
                     doc_weights,
                     alpha,
                     beta,
                     model=None,
                     all_policy_scores=None,
                     return_display=False,
                     store_per_rank=False,
                     deterministic_ranking=False):
  
  n_queries = data_split.num_queries()
  n_docs = data_split.num_docs()
  sampled_queries = np.random.choice(n_queries, size=n_samples)

  samples_per_query = np.zeros(n_queries, dtype=np.int32)
  np.add.at(samples_per_query, sampled_queries, 1)

  if all_policy_scores is None and n_samples > n_queries*0.8:
    all_policy_scores = model(data_split.feature_matrix)[:, 0].numpy()

  if store_per_rank:
    cutoff = alpha.shape[0]
    all_clicks = np.zeros((n_docs, cutoff), dtype=np.int64)
    all_displays = np.zeros((n_docs, cutoff), dtype=np.int64)
  else:
    all_clicks = np.zeros(n_docs, dtype=np.int64)
    all_displays = np.zeros(n_docs, dtype=np.int64)

  pair_ranges = np.zeros(n_queries + 1, dtype=np.int64)
  all_pairs = tuple([] for _ in range(5))

  prev_qid = 0
  for qid in np.arange(n_queries)[np.greater(samples_per_query, 0)]:
    q_clicks = data_split.query_values_from_vector(
                                  qid, all_clicks)
    q_displays = data_split.query_values_from_vector(
                                  qid, all_displays)
    (new_clicks,
     new_displays,
     new_pairwise) = single_query_generation(
                          qid,
                          data_split,
                          samples_per_query[qid],
                          doc_weights,
                          alpha,
                          beta,
                          model=model,
                          all_policy_scores=all_policy_scores,
                          return_display=return_display,
                          store_per_rank=store_per_rank,
                          deterministic_ranking=deterministic_ranking)

    q_clicks += new_clicks
    if store_per_rank:
      q_displays += new_displays
    else:
      q_displays += samples_per_query[qid]

    if new_pairwise is not None:
      pair_ranges[prev_qid+1:qid+1] = pair_ranges[prev_qid]
      pair_ranges[qid+1] = pair_ranges[prev_qid] + new_pairwise[0].shape[0]
      prev_qid = qid+1
      for i in range(len(all_pairs)):
        all_pairs[i].append(new_pairwise[i])
        # all_pairs = tuple(np.concatenate((all_pairs[i], new_pairwise[i])) for i in range(len(all_pairs)))

  all_pairs = tuple(np.concatenate(all_pairs[i]) for i in range(len(all_pairs)))
  pair_ranges[prev_qid:] = pair_ranges[prev_qid]
  if all_pairs == None:
    pairwise_info = None
  else:
    pairwise_info = (pair_ranges,) + all_pairs

  return all_clicks, all_displays, samples_per_query, pairwise_info

def single_query_generation(
                    qid,
                    data_split,
                    n_samples,
                    doc_weights,
                    alpha,
                    beta,
                    model=None,
                    all_policy_scores=None,
                    return_display=False,
                    store_per_rank=False,
                    deterministic_ranking=False):
  assert model is not None or policy_scores is not None

  n_docs = data_split.query_size(qid)
  cutoff = min(alpha.shape[0], n_docs)

  if all_policy_scores is None:
    q_feat = data_split.query_feat(qid)
    policy_scores = model(q_feat)[:,0].numpy()
  else:
    policy_scores = data_split.query_values_from_vector(
                                  qid, all_policy_scores)

  if deterministic_ranking:
    ranking = rnk.rank_and_invert(policy_scores)[0][:cutoff]
    rankings = np.tile(ranking[None, :], (n_samples, 1))
  else:
    rankings = pl.gumbel_sample_rankings(
                        policy_scores,
                        n_samples,
                        cutoff)[0]

  q_doc_weights = data_split.query_values_from_vector(
                                  qid, doc_weights)
  clicks = generate_clicks(
                          rankings,
                          q_doc_weights,
                          alpha, beta)

  pairwise_click_mask = np.logical_and(clicks[:, :, None], np.logical_not(clicks[:, None, :]))

  clicked_id = np.broadcast_to(rankings[:, :, None], (n_samples, cutoff, cutoff))[pairwise_click_mask]
  not_clicked_id = np.broadcast_to(rankings[:, None, :], (n_samples, cutoff, cutoff))[pairwise_click_mask]
  clicked_rank = np.broadcast_to(np.arange(cutoff)[None, :, None], (n_samples, cutoff, cutoff))[pairwise_click_mask]
  not_clicked_rank = np.broadcast_to(np.arange(cutoff)[None, None, :], (n_samples, cutoff, cutoff))[pairwise_click_mask]

  if store_per_rank:
    store_cutoff = alpha.shape[0]
    clicks_per_doc = np.zeros((n_docs, store_cutoff),
                              dtype=np.int32)
    ind_tile = np.tile(np.arange(cutoff)[None,:], (n_samples, 1))
    np.add.at(clicks_per_doc, (rankings[clicks],
                               ind_tile[clicks]), 1)
  else:
    clicks_per_doc = np.zeros(n_docs, dtype=np.int32)
    np.add.at(clicks_per_doc, rankings[clicks], 1)

  if return_display:
    if store_per_rank:
      displays_per_doc = np.zeros((n_docs, store_cutoff),
                                  dtype=np.int32)
      np.add.at(displays_per_doc, (rankings, ind_tile), 1)
    else:
      displays_per_doc = np.zeros(n_docs, dtype=np.int32)
      np.add.at(displays_per_doc, rankings, 1)
  else:
    displays_per_doc = None

  if clicked_id.shape[0] == 0:
    pairwise_click_info = None
  else:
    rank_i = np.lexsort((not_clicked_rank, clicked_rank, not_clicked_id, clicked_id))

    clicked_id = clicked_id[rank_i]
    not_clicked_id = not_clicked_id[rank_i]
    clicked_rank = clicked_rank[rank_i]
    not_clicked_rank = not_clicked_rank[rank_i]

    selected_counter = np.zeros_like(not_clicked_rank)
    selected_counter[0] = 1.

    selected_i = 1
    for i in range(1,rank_i.shape[0]):
      if (clicked_id[i] == clicked_id[i-1]
        and not_clicked_id[i] == not_clicked_id[i-1]
        and clicked_rank[i] == clicked_rank[i-1]
        and not_clicked_rank[i] == not_clicked_rank[i-1]):
        selected_counter[selected_i-1] += 1
      else:
        clicked_id[selected_i] = clicked_id[i]
        not_clicked_id[selected_i] = not_clicked_id[i]
        clicked_rank[selected_i] = clicked_rank[i]
        not_clicked_rank[selected_i] = not_clicked_rank[i]
        selected_counter[selected_i] = 1.
        selected_i += 1

    pairwise_click_info = (
        clicked_id[:selected_i],
        not_clicked_id[:selected_i],
        clicked_rank[:selected_i],
        not_clicked_rank[:selected_i],
        selected_counter[:selected_i],)

  return clicks_per_doc, displays_per_doc, pairwise_click_info

def generate_clicks(sampled_rankings,
                    doc_weights,
                    alpha, beta):
  cutoff = min(sampled_rankings.shape[1], alpha.shape[0])
  ranked_weights = doc_weights[sampled_rankings]
  click_prob = ranked_weights*alpha[None, :cutoff] + beta[None, :cutoff]

  noise = np.random.uniform(size=click_prob.shape)
  return noise < click_prob

def deterministic_simulate_on_dataset(
                        data_train,
                        data_validation,
                        n_samples,
                        train_doc_weights,
                        train_alpha,
                        train_beta,
                        validation_doc_weights,
                        vali_alpha,
                        vali_beta,
                        ):
  n_train_queries = data_train.num_queries()
  n_vali_queries = data_validation.num_queries()

  train_ratio = n_train_queries/float(n_train_queries + n_vali_queries)

  sampled_queries = np.random.choice(2, size=n_samples,
                                     p=[train_ratio, 1.-train_ratio])
  samples_per_split = np.zeros(2, dtype=np.int64)
  np.add.at(samples_per_split, sampled_queries, 1)

  (train_clicks,
   train_samples_per_query) = deterministic_simulate_queries(
                     data_train,
                     samples_per_split[0],
                     train_doc_weights,
                     train_alpha,
                     train_beta,)

  (validation_clicks,
   validation_samples_per_query) = deterministic_simulate_queries(
                     data_validation,
                     samples_per_split[0],
                     validation_doc_weights,
                     vali_alpha,
                     vali_beta,)

  return (train_clicks, train_samples_per_query,
          validation_clicks, validation_samples_per_query)

def deterministic_simulate_queries(data_split,
                     n_samples,
                     doc_weights,
                     doc_alpha,
                     doc_beta,):
  doc_click_prob = doc_alpha*doc_weights + doc_beta

  n_queries = data_split.num_queries()
  n_docs = data_split.num_docs()
  sampled_queries = np.random.choice(n_queries, size=n_samples)

  samples_per_query = np.zeros(n_queries, dtype=np.int32)
  np.add.at(samples_per_query, sampled_queries, 1)

  squared_matrix_indices = np.zeros(n_queries+1, dtype=np.int64)
  squared_matrix_indices[1:] = np.cumsum(data_split.query_sizes()**2)
  all_pairwise_pref = np.zeros(squared_matrix_indices[-1], dtype=np.float64)

  for qid in np.where(samples_per_query)[0]:
    q_click_prob = data_split.query_values_from_vector(qid, doc_click_prob)
    q_n_docs = q_click_prob.shape[0]
    if q_n_docs == 1:
      continue
    noise = np.random.uniform(size=(samples_per_query[qid], q_n_docs))
    q_clicks = np.less(noise, q_click_prob[None, :])
    q_pairwise_pref = np.sum(np.greater(q_clicks[:,:,None], q_clicks[:,None,:]), axis=0)

    s_i, e_i = squared_matrix_indices[qid:qid+2]
    all_pairwise_pref[s_i:e_i] = q_pairwise_pref.flatten()
    all_pairwise_pref[s_i:e_i] /= samples_per_query[qid]

  return all_pairwise_pref, samples_per_query