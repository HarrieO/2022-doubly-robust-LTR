# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import numpy as np
import utils.ranking as rnk

def gumbel_sample_rankings(log_scores, n_samples, cutoff=None, 
                           inverted=False, doc_prob=False,
                           prob_per_rank=False, return_gumbel=False,
                           return_full_rankings=False):
  n_docs = log_scores.shape[0]
  ind = np.arange(n_samples)

  if cutoff:
    ranking_len = min(n_docs, cutoff)
  else:
    ranking_len = n_docs

  if prob_per_rank:
    rank_prob_matrix = np.empty((ranking_len, n_docs), dtype=np.float64)

  gumbel_samples = np.random.gumbel(size=(n_samples, n_docs))
  gumbel_scores = log_scores[None,:]+gumbel_samples

  rankings, inv_rankings = rnk.multiple_cutoff_rankings(
                                -gumbel_scores,
                                ranking_len,
                                invert=inverted,
                                return_full_rankings=return_full_rankings)

  if not doc_prob:
    if not return_gumbel:
      return rankings, inv_rankings, None, None, None
    else:
      return rankings, inv_rankings, None, None, gumbel_scores

  log_scores = np.tile(log_scores[None,:], (n_samples, 1))
  rankings_prob = np.empty((n_samples, ranking_len), dtype=np.float64)
  for i in range(ranking_len):
    log_scores += 18 - np.amax(log_scores, axis=1)[:, None]
    log_denom = np.log(np.sum(np.exp(log_scores), axis=1))
    probs = np.exp(log_scores - log_denom[:, None])
    if prob_per_rank:
      rank_prob_matrix[i, :] = np.mean(probs, axis=0)
    rankings_prob[:, i] = probs[ind, rankings[:, i]]
    log_scores[ind, rankings[:, i]] = np.NINF

  if return_gumbel:
    gumbel_return_values = gumbel_scores
  else:
    gumbel_return_values = None

  if prob_per_rank:
    return rankings, inv_rankings, rankings_prob, rank_prob_matrix, gumbel_return_values
  else:
    return rankings, inv_rankings, rankings_prob, None, gumbel_return_values

def metrics_based_on_samples(sampled_rankings,
                             weight_per_rank,
                             addition_per_rank,
                             weight_per_doc,):
  cutoff = sampled_rankings.shape[1]
  return np.sum(np.mean(
              weight_per_doc[sampled_rankings]*weight_per_rank[None, :cutoff],
            axis=0) + addition_per_rank[:cutoff], axis=0)

def datasplit_metrics(data_split,
                      policy_scores,
                      weight_per_rank,
                      addition_per_rank,
                      weight_per_doc,
                      query_norm_factors=None,
                      n_samples=1000):
  cutoff = weight_per_rank.shape[0]
  n_queries = data_split.num_queries()
  results = np.zeros((n_queries, weight_per_rank.shape[1]),)
  for qid in range(n_queries):
    q_doc_weights = data_split.query_values_from_vector(qid, weight_per_doc)
    if not np.all(np.equal(q_doc_weights, 0.)):
      q_policy_scores = data_split.query_values_from_vector(qid, policy_scores)
      sampled_rankings = gumbel_sample_rankings(q_policy_scores,
                                                n_samples,
                                                cutoff=cutoff)[0]
      results[qid] = metrics_based_on_samples(sampled_rankings,
                                              weight_per_rank,
                                              addition_per_rank,
                                              q_doc_weights[:, None])
  if query_norm_factors is not None:
    results /= query_norm_factors

  return np.mean(results, axis=0)

def gradient_based_on_samples(rank_weights, labels, scores,
                              n_samples=None,
                              sampled_rankings=None):
  n_docs = labels.shape[0]
  result = np.zeros(n_docs, dtype=np.float64)
  cutoff = min(rank_weights.shape[0], n_docs)

  if n_docs == 1:
    return np.zeros_like(scores)

  scores = scores.copy() - np.amax(scores) + 10.

  assert n_samples is not None or sampled_rankings is not None
  if sampled_rankings is None:
    sampled_rankings = gumbel_sample_rankings(
                                    scores,
                                    n_samples,
                                    cutoff=cutoff,
                                    return_full_rankings=True)[0]
  else:
    n_samples = sampled_rankings.shape[0]

  cutoff_sampled_rankings = sampled_rankings[:,:cutoff]

  srange = np.arange(n_samples)

  relevant_docs = np.where(np.not_equal(labels, 0))[0]
  n_relevant_docs = relevant_docs.size

  weighted_labels = labels[cutoff_sampled_rankings]*rank_weights[None,:cutoff]
  cumsum_labels = np.cumsum(weighted_labels[:,::-1], axis=1)[:,::-1]

  np.add.at(result, cutoff_sampled_rankings[:,:-1], cumsum_labels[:,1:])
  result /= n_samples

  exp_scores = np.exp(scores).astype(np.float64)
  denom_per_rank = np.cumsum(exp_scores[sampled_rankings[:,::-1]], axis=1)[:,:-cutoff-1:-1]

  cumsum_weight_denom = np.cumsum(rank_weights[:cutoff]/denom_per_rank, axis=1)
  cumsum_reward_denom = np.cumsum(cumsum_labels/denom_per_rank, axis=1)  

  if cutoff < n_docs:
    second_part = -exp_scores[None,:]*cumsum_reward_denom[:,-1,None]
    second_part[:,relevant_docs] += (labels[relevant_docs][None,:]
        *exp_scores[None,relevant_docs]*cumsum_weight_denom[:,-1,None])
  else:
    second_part = np.empty((n_samples, n_docs), dtype=np.float64)

  sampled_direct_reward = labels[cutoff_sampled_rankings]*exp_scores[cutoff_sampled_rankings]*cumsum_weight_denom
  sampled_following_reward = exp_scores[cutoff_sampled_rankings]*cumsum_reward_denom
  second_part[srange[:,None], cutoff_sampled_rankings] = sampled_direct_reward - sampled_following_reward

  return result + np.mean(second_part, axis=0)
