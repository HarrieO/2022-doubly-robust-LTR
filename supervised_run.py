# Copyright (C) H.R. Oosterhuis 2022.
# Distributed under the MIT License (see the accompanying README.md and LICENSE files).

import argparse
import numpy as np
import time
import tensorflow as tf
import json

import utils.click_generation as clkgn
import utils.dataset as dataset
import utils.evaluation as evl
import utils.nnmodel as nn
import utils.optimization as opt

parser = argparse.ArgumentParser()
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

args = parser.parse_args()

click_model_name = args.click_model
cutoff = args.cutoff

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

max_ranking_size = np.min((cutoff, data.max_query_size()))

alpha, beta = clkgn.get_alpha_beta(click_model_name, cutoff)

model_params = {'hidden units': [32, 32],}

model = nn.init_model(model_params)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

first_results = evl.evaluate_policy(
                      model,
                      data.test,
                      data.test.label_vector*0.25,
                      alpha,
                      beta,
                    )

model, vali_reward = opt.optimize_policy(model, optimizer,
                      data_train=data.train,
                      train_doc_weights=data.train.label_vector*0.25,
                      train_alpha=alpha[:5]+beta[:5],
                      train_beta=np.zeros_like(beta[:5]),
                      data_vali=data.validation,
                      vali_doc_weights=data.validation.label_vector*0.25,
                      vali_alpha=alpha[:5]+beta[:5],
                      vali_beta=np.zeros_like(beta[:5]),
                      early_stop_per_epochs = 3,
                      print_updates=True,
                      max_epochs = 500,
                      )

final_results = evl.evaluate_policy(
                      model,
                      data.test,
                      data.test.label_vector*0.25,
                      alpha,
                      beta,
                    )

output = {
  'dataset': args.dataset,
  'fold number': args.fold_id,
  'click model': click_model_name,
  'initial model': 'random initialization',
  'run name': 'supervised run',
  'model hyperparameters': model_params,
  'results': final_results,
}

print(output)

print('Writing results to %s' % args.output_path)
with open(args.output_path, 'w') as f:
  json.dump(output, f)
