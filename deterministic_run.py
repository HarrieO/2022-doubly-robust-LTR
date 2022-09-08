# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import json
import numpy as np
import tensorflow as tf
import time

import utils.click_generation as clkgn
import utils.dataset as dataset
import utils.estimators as est
import utils.evaluation as evl
import utils.nnmodel as nn
import utils.optimization as opt
import utils.bias_estimation as EM
import utils.ranking as rnk

parser = argparse.ArgumentParser()
parser.add_argument("n_queries_sampled", type=int,
                    help="Number of sampled queries for training and validation.")
parser.add_argument("output_path", type=str,
                    help="Path to output model.")
parser.add_argument("--fold_id", type=int,
                    help="Fold number to select, modulo operator is applied to stay in range.",
                    default=1)
parser.add_argument("--click_model", type=str,
                    help="Name of click model to use.",
                    default='long')
parser.add_argument("--dataset", type=str,
                    default="Webscope_C14_Set1",
                    help="Name of dataset to sample from.")
parser.add_argument("--dataset_info_path", type=str,
                    default="local_dataset_info.txt",
                    help="Path to dataset info file.")
parser.add_argument("--pretrained_model", type=str,
                    default=None,
                    help="Path to pretrianed model file.")
parser.add_argument("--estimator", type=str,
                    default="DR",
                    help="Estimator to use: Naive/IPS/DM/DR.")
parser.add_argument('--quick_run', action='store_true',
                    help="Quick run for testing setting.")

args = parser.parse_args()

estimator = args.estimator

click_model_name = args.click_model
n_queries_sampled = args.n_queries_sampled

data = dataset.get_dataset_from_json_info(
                  args.dataset,
                  args.dataset_info_path,
                  shared_resource = False,
                )
fold_id = (args.fold_id-1)%data.num_folds()
data = data.get_data_folds()[fold_id]

start = time.time()
data.read_data()
print('Time past for reading data: %d seconds' % (time.time() - start))

cutoff = data.max_query_size()

alpha, beta = clkgn.get_alpha_beta(click_model_name, cutoff)

print('cutoff', cutoff)

print('True alpha', alpha)
print('True beta', beta)
print('alpha + beta', alpha + beta)

true_train_doc_weights = data.train.label_vector*0.25
true_vali_doc_weights = data.validation.label_vector*0.25
true_test_doc_weights = data.test.label_vector*0.25

model_params = {'hidden units': [32, 32],}
model = nn.init_model(model_params)
logging_model = nn.init_model(model_params)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model.build(input_shape=data.train.feature_matrix.shape)
logging_model.build(input_shape=data.train.feature_matrix.shape)
if args.pretrained_model:
  model.load_weights(args.pretrained_model)
  logging_model.load_weights(args.pretrained_model)
init_weights = model.get_weights()
init_opt_weights = optimizer.get_weights()

if estimator in ('IPS', 'Naive'):
  regression_model_params = None
  regression_model = None
  regression_optimizer = None
else:
  regression_model_params = {'hidden units': [32, 32], 'final activation': True}
  regression_model = nn.init_model(regression_model_params)
  regression_optimizer = tf.keras.optimizers.Adam()
  regression_model.build(input_shape=data.train.feature_matrix.shape)
  regression_init_weights = regression_model.get_weights()
  regression_init_opt_weights = regression_optimizer.get_weights()

train_clicks = np.zeros(data.train.num_docs())
train_query_freq = np.zeros(data.train.num_queries())
train_exp_alpha = np.zeros(data.train.num_docs())
train_exp_beta = np.zeros(data.train.num_docs())
train_policy_scores = logging_model(data.train.feature_matrix)[:, 0].numpy()

vali_clicks = np.zeros(data.validation.num_docs())
vali_query_freq = np.zeros(data.validation.num_queries())
vali_exp_alpha = np.zeros(data.validation.num_docs())
vali_exp_beta = np.zeros(data.validation.num_docs())
vali_policy_scores = logging_model(data.validation.feature_matrix)[:, 0].numpy()

test_policy_scores = logging_model(data.test.feature_matrix)[:, 0].numpy()

alpha_clip = min(100./np.sqrt(n_queries_sampled), 1.)
beta_clip = 0.

output = {
  'dataset': args.dataset,
  'fold number': args.fold_id,
  'click model': click_model_name,
  'initial model': 'random initialization',
  'run name': estimator,
  'model hyperparameters': model_params,
  'regression_model_params': regression_model_params,
  'bias estimation': 'given',
  'alpha clip': alpha_clip,
  'deterministic ranking': True,
}

if args.pretrained_model:
  output['initial model'] = args.pretrained_model

train_ranking = rnk.data_split_rank_and_invert(train_policy_scores, data.train)[1]
vali_ranking = rnk.data_split_rank_and_invert(vali_policy_scores, data.validation)[1]
test_ranking = rnk.data_split_rank_and_invert(test_policy_scores, data.test)[1]

logging_policy_metrics = evl.evaluate_deterministic_ranking(
                                test_ranking,
                                data.test,
                                true_test_doc_weights,
                                alpha,
                                beta
                              )

print('Alpha clip:', alpha_clip)

print('sampling up to %s queries' % n_queries_sampled)

true_train_alpha = alpha[train_ranking]
true_train_beta = beta[train_ranking]
true_vali_alpha = alpha[vali_ranking]
true_vali_beta = beta[vali_ranking]

(train_clicks, train_query_freq,
 vali_clicks, vali_query_freq,
     ) = clkgn.deterministic_simulate_on_dataset(
               data.train,
               data.validation,
               n_queries_sampled,
               true_train_doc_weights,
               true_train_alpha,
               true_train_beta,
               true_vali_doc_weights,
               true_vali_alpha,
               true_vali_beta,
               )
train_display_mask = np.greater(train_query_freq, 0)[data.train.query_index_per_document()]
vali_display_mask = np.greater(vali_query_freq, 0)[data.validation.query_index_per_document()]
obs_train_alpha = true_train_alpha*train_display_mask
obs_train_beta = true_train_beta*train_display_mask
obs_vali_alpha = true_vali_alpha*vali_display_mask
obs_vali_beta = true_vali_beta*vali_display_mask

print('done')

if estimator in ['DR', 'DM']:
  print('Optimize regression with an unbiased loss estimate.')
  regression_model.set_weights(regression_init_weights)
  opt.optimize_regressor_counterfactual_param_given(
                    regression_model, regression_optimizer,
                    data.train, train_query_freq, train_clicks[:, None], obs_train_alpha, obs_train_beta,
                    data.validation, vali_query_freq, vali_clicks[:, None], obs_vali_alpha, obs_vali_beta,
                    alpha_clip=alpha_clip
                    )

  
if estimator == 'DM':
  train_doc_weights = est.compute_direct_method_weights(
                          data.train,
                          train_query_freq,
                          regression_model,
                          )
  vali_doc_weights = est.compute_direct_method_weights(
                          data.validation,
                          vali_query_freq,
                          regression_model,
                          )
elif estimator == 'Naive':
  train_doc_weights = train_clicks/np.maximum(train_query_freq[data.train.query_index_per_document()],1.)
  vali_doc_weights = vali_clicks/np.maximum(vali_query_freq[data.validation.query_index_per_document()],1.)
elif estimator in ['DR', 'IPS']:
  train_doc_weights = est.compute_weights(
                            data.train,
                            train_clicks[:,None],
                            train_query_freq,
                            obs_train_alpha,
                            obs_train_beta,
                            alpha_clip=alpha_clip,
                            beta_clip=beta_clip,
                            regression_model=regression_model,
                            normalize=False,
                            )

  vali_doc_weights = est.compute_weights(
                            data.validation,
                            vali_clicks[:,None],
                            vali_query_freq,
                            obs_vali_alpha,
                            obs_vali_beta,
                            alpha_clip=alpha_clip,
                            beta_clip=beta_clip,
                            regression_model=regression_model,
                            normalize=False,
                            )
else:
  assert False, 'Unknown estimator: %s, allowed options are Naive/IPS/DM/DR' % estimator

if args.quick_run:
  max_epochs = 3
else:
  max_epochs = 500

model.set_weights(init_weights)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model, vali_reward = opt.optimize_policy(model, optimizer,
                      data_train=data.train,
                      train_doc_weights=train_doc_weights,
                      train_alpha=alpha+beta,
                      train_beta=np.zeros_like(beta),
                      data_vali=data.validation,
                      vali_doc_weights=vali_doc_weights,
                      vali_alpha=alpha+beta,
                      vali_beta=np.zeros_like(beta),
                      max_epochs=max_epochs,
                      early_stop_per_epochs=3,
                      print_updates=True,
                      )

cur_metrics = evl.evaluate_policy(
                      model,
                      data.test,
                      true_test_doc_weights,
                      alpha,
                      beta,
                    )

output['results'] = {
  'iteration': int(n_queries_sampled),
  'metrics': cur_metrics,
  'logging policy metrics': logging_policy_metrics,
}

print('No. query %09d, RCTR %0.5f, DCG %0.5f' % (
      n_queries_sampled, cur_metrics['RCTR'], cur_metrics['DCG']))

print(output)

print('Writing results to %s' % args.output_path)
with open(args.output_path, 'w') as f:
  json.dump(output, f)
