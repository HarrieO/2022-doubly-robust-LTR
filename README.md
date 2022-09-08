# Doubly-Robust Estimation for Correcting Position-Bias in Click Feedback for Unbiased Learning to Rank
This repository contains the code used for the experiments in "Doubly-Robust Estimation for Correcting Position-Bias in Click Feedback for Unbiased Learning to Rank" ([preprint available](https://harrieo.github.io//publication/2022-doubly-robust)).

Citation
--------

If you use this code to produce results for your scientific publication, or if you share a copy or fork, please refer to the pre-print paper:
```
@article{oosterhuis2022doubly,
  title={Doubly-Robust Estimation for Unbiased Learning-to-Rank from Position-Biased Click Feedback},
  author={Oosterhuis, Harrie},
  journal={arXiv preprint arXiv:2203.17118},
  year={2022}
}
```

License
-------

The contents of this repository are licensed under the [MIT license](LICENSE). If you modify its contents in any way, please link back to this repository.

Usage
-------

This code makes use of [Python 3](https://www.python.org/), the [numpy](https://numpy.org/) and the [tensorflow](https://www.tensorflow.org/) packages, make sure they are installed.

A file is required that explains the location and details of the LTR datasets available on the system, for the Yahoo! Webscope, MSLR-Web30k, and Istella datasets an example file is available. Copy the file:
```
cp example_datasets_info.txt local_dataset_info.txt
```
Open this copy and edit the paths to the folders where the train/test/vali files are placed.

Here are some command-line examples that illustrate how the results in the paper can be replicated.
First create a folder to store the resulting models:
```
mkdir local_output
```
The following command generates *N=100000* impressions in a top-5 setting on the Yahoo! dataset, and subsequently, uses the doubly robust estimator with known bias parameters to unbiasedly optimize a ranking policy:
```
python3 run.py 100000 local_output/IPS_top5_biasknown_N100000.txt --cutoff 5 --dataset Webscope_C14_Set1 --pretrained_model pretrained/Webscope_C14_Set1/pretrained_model.h5 --estimator DR
```
For other estimators change the *--estimator* flag, it accepts the following values *Naive*, *IPS*, *DM* or *DR*.
Adding the *--estimate_bias* flag makes bias parameters estimated instead of known.
To reproduce the ablation studies, use the *--clip_multiplier* and *--bias_interpolation* flags.

A different file can be used to reproduce the deterministic full-ranking setting:
```
python3 deterministic_run.py 100000 local_output/IPS_fullranking_biasknown_N100000.txt --estimator DR --cutoff 5 --dataset Webscope_C14_Set1 --pretrained_model pretrained/Webscope_C14_Set1/pretrained_model.h5
```
This file also accepts the *--estimator* flag as described above.

The Direct Method and Ratioprop baselines have seperate files. The following runs the baseline Direct Method with its linear bias model in the top-5 setting:
```
python3 linear_DM_run.py 100000 local_output/linearDM_top5_biasknown_N100000.txt --cutoff 5 --dataset Webscope_C14_Set1 --pretrained_model pretrained/Webscope_C14_Set1/pretrained_model.h5
```
Again there is a seperate file for the deterministic full-ranking setting:
```
python3 linear_DM_deterministic_run.py 100000 local_output/linearDM_fullranking_biasknown_N100000.txt --dataset Webscope_C14_Set1 --pretrained_model pretrained/Webscope_C14_Set1/pretrained_model.h5
```
Similarly, for the Ratioprop baseline:
```
python3 ratioprop_run.py 100000 local_output/ratioprop_top5_biasknown_N100000.txt --cutoff 5 --dataset Webscope_C14_Set1 --pretrained_model pretrained/Webscope_C14_Set1/pretrained_model.h5
```
```
python3 ratioprop_deterministic_run.py 100000 local_output/ratioprop_fullranking_biasknown_N100000.txt --dataset Webscope_C14_Set1 --pretrained_model pretrained/Webscope_C14_Set1/pretrained_model.h5
```