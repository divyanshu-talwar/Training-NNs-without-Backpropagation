# Training Neural Networks without Backpropagation

## Introduction
In this work we train neural networks by solving an optimization problem where the different layers are separated by variable splitting technique and the ensuing sub-problems are solved using alternating direction method of multipliers (ADMM). The performance of this optimization based alternative is compared with backpropagation (the traditional method to train neural networks) on the grounds of test/train accuracies, time to convergence and epochs run before convergence. We run our experiments on 10 regression and 10 classification datasets (20 datasets in total).

-	For technical issues, please report to the [Issues](https://github.com/divyanshu-talwar/Training-NNs-without-Backpropagation/issues) section.

## Dependencies
* Python (3.6.8):
    > cycler==`0.10.0`, decorator==`4.4.0`, matplotlib==`2.1.0`, networkx==`2.4`, numpy==`1.13.3`, Pillow==`6.2.1`, pyparsing==`2.4.2`, python-dateutil==`2.8.0`, pytz==`2019.3`, PyWavelets==`1.1.1`, PyYAML==`5.1.2`, scikit-image==`0.13.0`, scikit-learn==`0.19.1`, scipy==`0.19.1`, six==`1.12.0`, torch==`0.3.1`, torchvision==`0.2.0`

_**Note:** All the aforementioned dependencies can be easily installed by executing the following command:_

`pip install -r requirements.txt`

## Contents
* `scripts/` - contains all the relevant scripts to run the experiment.
	* `scripts/backprop_implementation.py` - implementation of backpropagation based training.
	* `scripts/admm_implementation.py` - implementation of ADMM based training.
	* `scripts/read_datasets.py` - contains classes for reading the datasets
	* `scripts/runall.sh` - bash script to run all the experiments mentioned in the paper.
* `datasets` - contains the `classification` and `regression` datasets downloaded from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php?format=&task=reg&att=num&area=&numAtt=&numIns=greater1000&type=&sort=nameUp&view=table).
* `results/` - contains the `accuracies`, training `logs` and loss vs epoch `plots` for both the approaches.

## Results
* Accuracies:
	* [Regression datasets](https://github.com/divyanshu-talwar/Training-NNs-without-Backpropagation/blob/master/results/regression_accuracies.pdf)
	* [Classification datasets](https://github.com/divyanshu-talwar/Training-NNs-without-Backpropagation/blob/master/results/classification_accuracies.pdf)
* Plots:
	* [Backpropagation](https://github.com/divyanshu-talwar/Training-NNs-without-Backpropagation/tree/master/results/backpropagation/plots)
	* [ADMM based](https://github.com/divyanshu-talwar/Training-NNs-without-Backpropagation/tree/master/results/admm/plots)
* Training Logs:
	* [Backpropagation](https://github.com/divyanshu-talwar/Training-NNs-without-Backpropagation/tree/master/results/backpropagation/logs)
	* [ADMM based](https://github.com/divyanshu-talwar/Training-NNs-without-Backpropagation/tree/master/results/admm/logs)

## Execution
* Installing the requirements is a pre-requisite to running the experiment. This can be achieved by using the following command:

`pip install -r requirements.txt`

* The following optional command-line arguments can be supplied to the scripts. For more information on these run `<script_name> -h` or `<script_name> --help`.
```
Options :
usage: backprop_implementation.py [-h] [--log_directory LOG_DIRECTORY]
                                  [--log_file LOG_FILE]
                                  [--plot_directory PLOT_DIRECTORY]
                                  [--plot_file PLOT_FILE] [--dataset DATASET]
                                  [--max_epoch MAX_EPOCH]
                                  [--initial_learning_rate INITIAL_LEARNING_RATE]
                                  [--threshold THRESHOLD]
                                  [--hidden_units HIDDEN_UNITS]
                                  [--is_classification IS_CLASSIFICATION]

usage: admm_implementation.py [-h]
                                              [--log_directory LOG_DIRECTORY]
                                              [--log_file LOG_FILE]
                                              [--plot_directory PLOT_DIRECTORY]
                                              [--plot_file PLOT_FILE]
                                              [--dataset DATASET]
                                              [--max_epoch MAX_EPOCH]
                                              [--threshold THRESHOLD]
                                              [--hidden_units HIDDEN_UNITS]
                                              [--is_classification IS_CLASSIFICATION]
                                              [--mu MU]

```
_**Note:** If you use any other dataset apart from the ones in here add the dataset to the datasets folder, write it's corresponding loader in `read_datasets.py` and run either/both the scripts by using the following command:_

`python <scipt-name> --dataset <dataset_name> --is_classification {True, False}`

* To run all the experiments and to reproduce the results simply execute `scripts/runall.sh` by using the following command:
```bash
bash runall.sh
```
