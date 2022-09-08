# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import utils.plackettluce as pl
import utils.ranking as rnk

def get_alpha_beta(click_model_name, cutoff):
  ranks = np.arange(cutoff)
  if click_model_name == 'default':
    pos_bias = 0.35*1/(ranks+1.) + 0.65/(1.+0.05*ranks)
    eplus = 1./(1.+0.005*ranks)
    emin = 0.65/(ranks+1.)
    return pos_bias*(eplus - emin), pos_bias*emin
  elif click_model_name == 'long':
    pos_bias = 1./(1 + ranks/5.)**2.
    eplus = 1.
    emin = 0.1 + 0.6/((ranks+1)/20.+1.)
    return pos_bias*(eplus - emin), pos_bias*emin
  else:
    raise NotImplementedError('Click model %s is not implemented' % click_model_name)

def deterministic_simulate_on_dataset(
                        data_train,
                        data_validation,
                        n_samples,
                        train_doc_weights,
                        train_alpha,
                        train_beta,
                        vali_doc_weights,
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
                     train_beta)

  (validation_clicks,
   validation_samples_per_query) = deterministic_simulate_queries(
                     data_validation,
                     samples_per_split[1],
                     vali_doc_weights,
                     vali_alpha,
                     vali_beta)

  return (train_clicks, train_samples_per_query,
          validation_clicks, validation_samples_per_query)

def deterministic_simulate_queries(
                    data_split,
                    n_samples,
                    doc_weights,
                    doc_alpha,
                    doc_beta,
                    ):
  
  n_queries = data_split.num_queries()
  n_docs = data_split.num_docs()
  sampled_queries = np.random.choice(n_queries, size=n_samples)

  samples_per_query = np.zeros(n_queries, dtype=np.int32)
  np.add.at(samples_per_query, sampled_queries, 1)

  all_clicks = np.zeros(n_docs, dtype=np.int64)
  for qid in np.arange(n_queries)[np.greater(samples_per_query, 0)]:
    q_clicks = data_split.query_values_from_vector(
                                  qid, all_clicks)
    q_alpha = data_split.query_values_from_vector(
                                  qid, doc_alpha)
    q_beta = data_split.query_values_from_vector(
                                  qid, doc_beta)
    q_weights = data_split.query_values_from_vector(
                                  qid, doc_weights)
    q_prob = q_alpha*q_weights + q_beta

    noise = np.random.uniform(size=(samples_per_query[qid], q_clicks.shape[0]))
    q_clicks[:] = np.sum(noise < q_prob[None, :], axis=0)
    
  return all_clicks, samples_per_query

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
   train_samples_per_query) = simulate_queries(
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
   validation_samples_per_query) = simulate_queries(
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
          train_samples_per_query,
          validation_clicks, validation_displays,
          validation_samples_per_query)


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
  for qid in np.arange(n_queries)[np.greater(samples_per_query, 0)]:
    q_clicks = data_split.query_values_from_vector(
                                  qid, all_clicks)
    q_displays = data_split.query_values_from_vector(
                                  qid, all_displays)
    (new_clicks,
     new_displays) = single_query_generation(
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

  return all_clicks, all_displays, samples_per_query

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

  return clicks_per_doc, displays_per_doc

def single_ranking_generation(
                     qid,
                     data_split,
                     doc_weights,
                     alpha,
                     beta,
                     model=None,
                     all_policy_scores=None,
                     return_scores=False):
  assert model is not None or policy_scores is not None

  n_docs = data_split.query_size(qid)
  cutoff = min(alpha.shape[0], n_docs)

  if all_policy_scores is None:
    q_feat = data_split.query_feat(qid)
    policy_scores = model(q_feat)[:,0].numpy()
  else:
    policy_scores = data_split.query_values_from_vector(
                                  qid, all_policy_scores)

  rankings = pl.gumbel_sample_rankings(
                      policy_scores,
                      1,
                      cutoff)[0]


  q_doc_weights = data_split.query_values_from_vector(
                                  qid, doc_weights)
  clicks = generate_clicks(
                          rankings,
                          q_doc_weights,
                          alpha, beta)

  if return_scores:
    return rankings[0,:], clicks[0,:], policy_scores
  else:
    return rankings[0,:], clicks[0,:]

def generate_clicks(sampled_rankings,
                    doc_weights,
                    alpha, beta):
  cutoff = min(sampled_rankings.shape[1], alpha.shape[0])
  ranked_weights = doc_weights[sampled_rankings]
  click_prob = ranked_weights*alpha[None, :cutoff] + beta[None, :cutoff]

  noise = np.random.uniform(size=click_prob.shape)
  return noise < click_prob
