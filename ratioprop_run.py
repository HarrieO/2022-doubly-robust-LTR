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
import utils.bias_estimation as EM
import utils.ratioprop as ratioprop

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
                    default='default')
parser.add_argument("--dataset", type=str,
                    default="Webscope_C14_Set1",
                    help="Name of dataset to sample from.")
parser.add_argument("--dataset_info_path", type=str,
                    default="local_dataset_info.txt",
                    help="Path to dataset info file.")
parser.add_argument("--cutoff", type=int,
                    help="Maximum number of items that can be displayed.",
                    default=5)
parser.add_argument("--pretrained_model", type=str,
                    default=None,
                    help="Path to pretrianed model file.")
parser.add_argument("--estimate_bias", action='store_true',
                    help="Flag to make bias estimated instead of given.")
parser.add_argument('--quick_run', action='store_true',
                    help="Quick run for testing setting.")

args = parser.parse_args()

bias_given = not args.estimate_bias

click_model_name = args.click_model
cutoff = args.cutoff
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

cutoff = min(cutoff, data.max_query_size())

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

n_sampled = 0

train_clicks = np.zeros((data.train.num_docs(), cutoff))
train_displays = np.zeros((data.train.num_docs(), cutoff))
train_query_freq = np.zeros(data.train.num_queries())
train_policy_scores = logging_model(data.train.feature_matrix)[:, 0].numpy()

vali_clicks = np.zeros((data.validation.num_docs(), cutoff))
vali_displays = np.zeros((data.validation.num_docs(), cutoff))
vali_query_freq = np.zeros(data.validation.num_queries())
vali_exp_alpha = np.zeros(data.validation.num_docs())
vali_exp_beta = np.zeros(data.validation.num_docs())
vali_policy_scores = logging_model(data.validation.feature_matrix)[:, 0].numpy()

alpha_clip = min(10./np.sqrt(n_queries_sampled), 1.)
beta_clip = 0.

print('Alpha clip:', alpha_clip)

output = {
  'dataset': args.dataset,
  'fold number': args.fold_id,
  'click model': click_model_name,
  'initial model': 'random initialization',
  'run name': 'ratio propensity scoring',
  'model hyperparameters': model_params,
  'regression_model_type': 'none',
  'regression_model_params': None,
  'bias estimated': not bias_given,
  'alpha clip': alpha_clip,
}

if args.pretrained_model:
  output['initial model'] = args.pretrained_model

logging_policy_metrics = evl.evaluate_policy(
                                model,
                                data.test,
                                true_test_doc_weights,
                                alpha,
                                beta,
                              )

print('sampling up to %s queries' % n_queries_sampled)

(new_train_clicks, new_train_displays,
 new_train_query_freq, train_pairwise_info,
 new_vali_clicks, new_vali_displays,
 new_vali_query_freq, vali_pairwise_info
  ) = ratioprop.simulate_on_dataset(
                         data.train,
                         data.validation,
                         n_queries_sampled - n_sampled,
                         true_train_doc_weights,
                         true_vali_doc_weights,
                         alpha,
                         beta,
                         model=logging_model,
                         train_policy_scores=train_policy_scores,
                         vali_policy_scores=vali_policy_scores,
                         store_per_rank=True,
                         return_display=True,
                         )
print('done')

if bias_given:
  est.update_observations(
                    train_clicks,
                    train_displays,
                    train_query_freq,
                    new_train_clicks,
                    new_train_displays,
                    new_train_query_freq,
                   )

  est.update_observations(
                    vali_clicks,
                    vali_displays,
                    vali_query_freq,
                    new_vali_clicks,
                    new_vali_displays,
                    new_vali_query_freq,
                   )

  if args.dataset == 'istella' and args.cutoff == 5:
    print('Using stored bias values')
    est_alpha = np.array([0.99999966, 0.65294275, 0.49920038, 0.41037026, 0.34778024], dtype=alpha.dtype)
  else:
    bias_n_queries = max(10**9, n_queries_sampled)
    print('sampling another %s queries for bias estimation' % max(bias_n_queries - n_queries_sampled, 0))

    (b_train_clicks, b_train_displays,
     b_train_query_freq,
     b_vali_clicks, b_vali_displays,
     b_vali_query_freq
     ) = clkgn.simulate_on_dataset(
                             data.train,
                             data.validation,
                             bias_n_queries - n_queries_sampled,
                             true_train_doc_weights,
                             true_vali_doc_weights,
                             alpha,
                             beta,
                             model=logging_model,
                             train_policy_scores=train_policy_scores,
                             vali_policy_scores=vali_policy_scores,
                             store_per_rank=True,
                             return_display=True,
                             )
    print('done')

    est.update_observations(
                      b_train_clicks,
                      b_train_displays,
                      b_train_query_freq,
                      train_clicks.astype(b_train_clicks.dtype),
                      train_displays.astype(b_train_displays.dtype),
                      train_query_freq.astype(b_train_query_freq.dtype),
                     )

    est.update_observations(
                      b_vali_clicks,
                      b_vali_displays,
                      b_vali_query_freq,
                      vali_clicks.astype(b_vali_clicks.dtype),
                      vali_displays.astype(b_vali_displays.dtype),
                      vali_query_freq.astype(b_vali_query_freq.dtype),
                     )

    print('Bias Estimation from A Very Large Sample')


    est_alpha = EM.EM_estimate_linear_bias(
                                data.train,
                                data.validation,
                                b_train_clicks,
                                b_train_displays,
                                b_vali_clicks,
                                b_vali_displays,
                                )
    print('done')

    del b_train_clicks
    del b_train_displays
    del b_vali_clicks
    del b_vali_displays

else:
  est.update_observations(
                    train_clicks,
                    train_displays,
                    train_query_freq,
                    new_train_clicks,
                    new_train_displays,
                    new_train_query_freq,
                   )

  est.update_observations(
                    vali_clicks,
                    vali_displays,
                    vali_query_freq,
                    new_vali_clicks,
                    new_vali_displays,
                    new_vali_query_freq,
                   )

  est_alpha = EM.EM_estimate_linear_bias(
                              data.train,
                              data.validation,
                              train_clicks,
                              train_displays,
                              vali_clicks,
                              vali_displays,
                              )

  print('done')

print('computing pair weights')
train_pair_weight_info = ratioprop.pairwise_weights_from_info(
                                      data.train, est_alpha, 1./alpha_clip,
                                      train_pairwise_info)
vali_pair_weight_info = ratioprop.pairwise_weights_from_info(
                                      data.validation, est_alpha, 1./alpha_clip,
                                      vali_pairwise_info)

del train_pairwise_info
del vali_pairwise_info

print('done')

if args.quick_run:
  max_epochs = 3
else:
  max_epochs = 500
model.set_weights(init_weights)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
model, vali_reward = ratioprop.optimize_policy(model, optimizer, est_alpha,
                    data_train=data.train,
                    train_pairwise_info=train_pair_weight_info,
                    data_vali=data.validation,
                    vali_pairwise_info=vali_pair_weight_info,
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
                      # multiplier turns the model into a deterministic ranker
                      model_output_multiplier=1000.,
                    )

output['results'] = {
  'iteration': int(n_queries_sampled),
  'metrics': cur_metrics,
  'logging policy metrics': logging_policy_metrics,
  'estimated alpha': list(est_alpha),
}

print('No. query %09d, RCTR %0.5f, DCG %0.5f' % (
      n_queries_sampled, cur_metrics['RCTR'], cur_metrics['DCG']))

print(output)

print('Writing results to %s' % args.output_path)
with open(args.output_path, 'w') as f:
  json.dump(output, f)

