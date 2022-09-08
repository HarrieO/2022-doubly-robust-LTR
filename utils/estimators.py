# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import utils.plackettluce as pl

def prob_per_rank_query(n_samples, cutoff, policy_scores):
  n_docs = policy_scores.size
  q_cutoff = min(cutoff, policy_scores.size)
  if n_docs <= 1:
    return np.ones((n_docs, q_cutoff))
  rankings = pl.gumbel_sample_rankings(
                      policy_scores,
                      n_samples,
                      q_cutoff)[0]
  freq_per_rank = np.zeros((n_docs, q_cutoff), dtype=np.int64)
  np.add.at(freq_per_rank, (rankings[:,:-1], np.arange(q_cutoff-1)[None,:]), 1)
  prob_per_rank = freq_per_rank.astype(np.float64)/n_samples

  scores_per_ranking = np.tile(policy_scores, (n_samples, 1)).astype(np.float64)
  scores_per_ranking[np.arange(n_samples)[:, None], rankings[:,:-1]] = np.NINF
  scores_per_ranking -= np.amax(scores_per_ranking, axis=1)[:, None]
  denom = np.log(np.sum(np.exp(scores_per_ranking), axis=1))[:, None]

  prob_per_rank[:, -1] = np.mean(np.exp(scores_per_ranking - denom), axis=0)

  return prob_per_rank

def expected_alpha_beta(
                    data_split,
                    n_samples,
                    alpha,
                    beta,
                    model=None,
                    all_policy_scores=None):
  if all_policy_scores is None:
    all_policy_scores = model(data_split.feature_matrix)[:,0].numpy()

  n_docs = data_split.num_docs()
  expected_alpha = np.zeros(n_docs, dtype=np.float64)
  expected_beta = np.zeros(n_docs, dtype=np.float64)
  for qid in range(data_split.num_queries()):
    cutoff = min(alpha.shape[0],
                 data_split.query_size(qid))

    q_alpha = data_split.query_values_from_vector(
                                  qid, expected_alpha)
    q_beta = data_split.query_values_from_vector(
                                  qid, expected_beta)
    policy_scores = data_split.query_values_from_vector(
                                  qid, all_policy_scores)

    q_prob_per_rank = prob_per_rank_query(n_samples,
                                          cutoff,
                                          policy_scores)

    q_alpha[:] = np.sum(q_prob_per_rank*alpha[None, :cutoff], axis=1)
    q_beta[:] = np.sum(q_prob_per_rank*beta[None, :cutoff], axis=1)

    assert np.all(np.greater(q_alpha, 0)), 'Zero alpha: %s' % q_alpha

  return expected_alpha, expected_beta

def compute_direct_method_weights(
                            data_split,
                            query_freq,
                            regression_model,
                            ):
  q_d_i = data_split.query_index_per_document()
  doc_mask = np.greater(query_freq, 0)[q_d_i]

  selected_feat = data_split.feature_matrix[doc_mask, :]
  result = np.zeros(data_split.num_docs())
  result[doc_mask] = regression_model(selected_feat)[:, 0].numpy()

  return result

def compute_weights(data_split,
                    clicks,
                    query_freq,
                    alpha_per_doc,
                    beta_per_doc,
                    normalize=False,
                    alpha_clip=None,
                    beta_clip=None,
                    regression_model=None,
                    ):
  if alpha_clip is None:
    clipped_alpha = alpha_per_doc
  else:
    clipped_alpha = np.maximum(alpha_per_doc, alpha_clip)
  if beta_clip is None:
    clipped_beta = beta_per_doc
  else:
    clipped_beta = np.maximum(beta_per_doc, beta_clip)
  
  clicks = np.sum(clicks, axis=1)
  q_d_i = data_split.query_index_per_document()
  q_mask = np.greater(query_freq, 0)
  q_included = np.sum(q_mask)
  if regression_model is None:
    safe_denom = clipped_alpha + np.equal(clipped_alpha, 0.)
    ctr = clicks / np.maximum(query_freq, 1.)[q_d_i]
    weights = (ctr-clipped_beta)/safe_denom
  else:
    if q_included < data_split.num_queries():
      doc_mask = q_mask[q_d_i]
      selected_feat = data_split.feature_matrix[doc_mask,:]
      regression_predictions = regression_model(selected_feat)[:, 0].numpy()

      selected_alpha = clipped_alpha[doc_mask]
      safe_query_freq = np.maximum(query_freq, 1.)

      safe_denom = selected_alpha + np.equal(selected_alpha, 0.).astype(np.float64)

      ctr = clicks[doc_mask]/safe_query_freq[q_d_i[doc_mask]]

      selected_weights = (ctr - clipped_beta[doc_mask]
        - (alpha_per_doc[doc_mask] - selected_alpha)*regression_predictions
        )/safe_denom

      weights = np.zeros(data_split.num_docs(), dtype=np.float64)
      weights[doc_mask] = selected_weights
    else:
      safe_denom = clipped_alpha + np.equal(clipped_alpha, 0.).astype(np.float64)
      safe_query_freq = np.maximum(query_freq, 1.)
      regression_predictions = regression_model(data_split.feature_matrix)[:, 0].numpy()
      ctr = clicks/safe_query_freq[q_d_i].astype(np.float64)

      weights = (ctr - clipped_beta
                  - (alpha_per_doc - clipped_alpha)*regression_predictions
                  )/safe_denom

  if len(weights.shape) == 2:
    weights = np.sum(weights, axis=0)

  if normalize:
    weights /= float(q_included/data_split.num_queries())

  return weights

def update_weights(data_split,
                   prev_clicks,
                   prev_displays,
                   prev_query_freq,
                   prev_exp_alpha,
                   prev_exp_beta,
                   new_clicks,
                   new_displays,
                   new_query_freq,
                   new_exp_alpha,
                   new_exp_beta,
                   n_samples,
                   alpha,
                   beta,
                   model=None,
                   all_policy_scores=None,
                   alpha_clip=None,
                   beta_clip=None,
                   regression_model=None):

  prev_n_sampled_queries = np.sum(prev_query_freq)
  new_n_sampled_queries = np.sum(new_query_freq)
  total_queries = prev_n_sampled_queries + new_n_sampled_queries
  prev_weight = prev_n_sampled_queries / total_queries
  new_weight = new_n_sampled_queries / total_queries

  prev_clicks += new_clicks
  prev_displays += new_displays
  prev_query_freq += new_query_freq

  prev_exp_alpha *= prev_weight
  prev_exp_alpha += new_weight*new_exp_alpha
  prev_exp_beta *= prev_weight
  prev_exp_beta += new_weight*new_exp_beta

  doc_weights = compute_weights(
                    data_split,
                    prev_exp_alpha,
                    prev_exp_beta,
                    prev_clicks,
                    prev_displays,
                    normalize=True,
                    n_queries_sampled=np.sum(prev_query_freq),
                    alpha_clip=alpha_clip,
                    beta_clip=beta_clip,
                    regression_model=regression_model,
                    alpha=alpha,
                    query_freq=prev_query_freq,
                  )

  return doc_weights

def update_statistics(
                   frequency_estimate,
                   data_split,
                   prev_clicks,
                   prev_displays,
                   prev_query_freq,
                   prev_exp_alpha,
                   prev_exp_beta,
                   new_clicks,
                   new_displays,
                   new_query_freq,
                   new_exp_alpha,
                   new_exp_beta,
                   alpha,
                   beta,
                   ):

  if not frequency_estimate:
    prev_n_sampled_queries = np.sum(prev_query_freq)
    new_n_sampled_queries = np.sum(new_query_freq)
    total_queries = prev_n_sampled_queries + new_n_sampled_queries
    prev_weight = prev_n_sampled_queries / total_queries
    new_weight = new_n_sampled_queries / total_queries

    prev_exp_alpha *= prev_weight
    prev_exp_alpha += new_weight*new_exp_alpha
    prev_exp_beta *= prev_weight
    prev_exp_beta += new_weight*new_exp_beta

  prev_clicks += new_clicks
  prev_displays += new_displays
  prev_query_freq += new_query_freq

  if frequency_estimate:
    safe_query_freq = np.maximum(prev_query_freq, 1.)
    d_q_i = data_split.query_index_per_document()
    display_prob = prev_displays/safe_query_freq[d_q_i, None]
    prev_exp_alpha[:] = np.sum(display_prob*alpha[None,:], axis=1)
    prev_exp_beta[:] = np.sum(display_prob*beta[None,:], axis=1)
  
def update_observations(
                   prev_clicks,
                   prev_displays,
                   prev_query_freq,
                   new_clicks,
                   new_displays,
                   new_query_freq,
                   ):

  prev_clicks += new_clicks
  prev_displays += new_displays
  prev_query_freq += new_query_freq

def update_frequency_estimated_bias(
                   data_split,
                   displays,
                   query_freq,
                   exp_alpha,
                   exp_beta,
                   alpha,
                   beta,
                   ):

  safe_query_freq = np.maximum(query_freq, 1.)
  d_q_i = data_split.query_index_per_document()
  display_prob = displays/safe_query_freq[d_q_i, None]
  exp_alpha[:] = np.sum(display_prob*alpha[None,:], axis=1)
  exp_beta[:] = np.sum(display_prob*beta[None,:], axis=1)

