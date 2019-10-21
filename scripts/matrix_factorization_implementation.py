import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as plt
import os
import timeit
import argparse
from read_datasets import read_classification_datasets, read_regression_datasets

plt.switch_backend('agg')

# Set a random seed (=Answer to the Ultimate Question of Life, the Universe, and Everything)
torch.manual_seed(42)
parser = argparse.ArgumentParser()

parser.add_argument('--log_directory', type=str, default='../results/matrix_factorization/logs', help="destination directory to save the training logs")
parser.add_argument('--log_file', type=str, default='boston_backprop.log', help="text filename to save training logs")
parser.add_argument('--plot_directory', type=str, default='../results/matrix_factorization/plots', help="destination directory to save the loss vs iteration plots")
parser.add_argument('--plot_file', type=str, default='boston_backprop.png', help=".png filename to save the loss vs iteration plots")
parser.add_argument('--dataset', type=str, default='boston', help="dataset to run the script on")

parser.add_argument('--max_epoch', type=int, default=15000, help="flag to indicate the maximum epochs for training")
parser.add_argument('--threshold', type=float, default=1e-3, help="less than threshold value change in subsequent losses implies convergence")
parser.add_argument('--hidden_units', type=int, default=100, help="number of units in the hidden layer of the neural network")
parser.add_argument('--is_classification', type=bool, default=False, help="Boolean to indicate if the problem at hand is a classification problem.")
parser.add_argument('--mu', type=float, default=1., help="hyper-parameter value")

FLAGS = parser.parse_args()

if not os.path.exists(FLAGS.log_directory):
		os.makedirs(FLAGS.log_directory)
		print("[!info] Successfully created log directory.")

if not os.path.exists(FLAGS.plot_directory):
		os.makedirs(FLAGS.plot_directory)
		print("[!info] Successfully created plots directory.")

log_file = os.path.join(FLAGS.log_directory, FLAGS.log_file)
plot_save = os.path.join(FLAGS.plot_directory, FLAGS.plot_file)

if FLAGS.is_classification:
	loaded_dataset = read_classification_datasets(FLAGS.dataset)
	features, targets = loaded_dataset.features, loaded_dataset.targets

	input_dim = features.shape[1]
	num_classes = 1

	labels = []
	num_classes = np.unique(targets).shape[0]
	if FLAGS.dataset == 'glass':
		num_classes = 7
	for i in targets:
		temp = np.zeros(num_classes)
		temp[i] = 1
		labels.append(temp)

	labels = np.array(labels)
	features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25, random_state=42)

	print('[!info] Datapoints = (Total: {0}) =  ((Test: {1}) + (Train: {2}))'.format(len(labels), len(labels_test), len(labels_train)))
	print("----")
	print()

	print('[!info] Input feature vector dimensionality: {0}'.format(input_dim))
	print('[!info] Hidden units: {0}'.format(FLAGS.hidden_units))
	print('[!info] Input dataset classes: {0}'.format(num_classes))
	print('[!info] Threshold : {0}'.format(FLAGS.threshold))
	print('[!info] mu : {0}'.format(FLAGS.mu))
	print('[!info] Maximum Iterations : {0}'.format(FLAGS.max_epoch))
	print("----")
	print()

	class Classifier(nn.Module):
		def __init__(self):
			super(Classifier, self).__init__()
			self.fc1 = nn.Linear(in_features=input_dim, out_features=FLAGS.hidden_units, bias=False)
			self.fc2 = nn.Linear(in_features=FLAGS.hidden_units, out_features=num_classes, bias=False)


		def forward(self, z):
			y = F.relu(self.fc1(z))
			x = self.fc2(y)
			return x

	def relu(inp):
	  return inp.clip(min=0)

	def forward_pass(W1, W2, inp):
	  output = relu(W2*relu(W1*inp))
	  return output.T

	model = Classifier()

	W1, W2 = np.matrix(list(model.parameters())[0].data.numpy()), np.matrix(list(model.parameters())[1].data.numpy())

	features_train_var, features_test_var, labels_train_var, labels_test_var = np.matrix(features_train).T, np.matrix(features_test).T, np.matrix(labels_train).T, np.matrix(labels_test).T

	with open(log_file, 'w') as log:
		log.write('Epoch\tLoss\n')

	start_time = timeit.default_timer()

	losses = []
	mu = 1.
	prev_loss = 0.
	H = relu(np.matrix(W1*features_train_var))

	# Actually runs for max_epoch number of times. Since loss in the first iteration is just the result of the random wt. matrix.
	for epoch in range(0, FLAGS.max_epoch):
		loss = np.power(np.linalg.norm(labels_train_var - forward_pass(W1, W2, features_train_var).T), 2)
		losses.append(loss)
		if(abs(loss - prev_loss) < FLAGS.threshold):
			print('[Results] Threshold value reached at iteration: {0}'.format(epoch))
			break
		prev_loss = loss
	  
		# Matrix factorization step
		W1 = H*np.linalg.pinv(features_train_var)
		W2 = labels_train_var*np.linalg.pinv(H)
		H = np.linalg.inv(W2.T*W2 + mu*np.identity(FLAGS.hidden_units)) * (W2.T*labels_train_var + mu*W1*features_train_var)
		H = H.clip(min = 0)
	  
		with open(log_file, 'a') as log:
			log.write('{0}\t{1}\n'.format(
				epoch,
				loss        
			))

		if (epoch + 1) % 5 == 0 or (epoch + 1) == FLAGS.max_epoch:    
			print("Loss for epoch {0} : {1} ".format(epoch, loss))
			print('')

	stop_time = timeit.default_timer()
	total_time = stop_time - start_time

	print("[Results] Total time in seconds: {0}".format(total_time))

	losses = np.array(losses)
	plt.plot(losses)
	plt.xlabel('Epoch')
	plt.ylabel('Loss ')
	plt.title('{0}'.format(FLAGS.dataset.replace('-', ' ').capitalize()))
	plt.locator_params(axis='y', nbins=12)

	plt.savefig(plot_save)

	## Calculate the Train Data Accuracy
	predictions = []

	predictions = np.array(forward_pass(W1, W2, features_train_var))
		
	score = 0
	for num in range(len(predictions)):
		if(np.argmax(labels_train[num]) == np.argmax(predictions[num])):
			score += 1
	accuracy = float(score)/float(len(predictions))
	print("[Results] Accuracy Train = {0}".format(accuracy * 100.))

	## Calculate the Test Data Accuracy
	predictions = []

	predictions = np.array(forward_pass(W1, W2, features_test_var))

	score = 0
	for num in range(len(predictions)):
		if(np.argmax(labels_test[num]) == np.argmax(predictions[num])):
			score += 1
	accuracy = float(score)/float(len(predictions))
	print("Accuracy Test = {0}".format(accuracy * 100.))

else:
	loaded_dataset = read_regression_datasets(FLAGS.dataset)
	features, targets = loaded_dataset.features, loaded_dataset.targets

	input_dim = features.shape[1]

	if type(targets[0]) == np.float64 or type(targets[0]) == np.int64:
		num_predicted_values = 1
	else:
		num_predicted_values = targets.shape[1]

	labels = []
	if(len(targets.shape) == 1):
		targets = np.reshape(targets, (targets.shape[0], 1))
	labels = targets

	labels = np.array(labels)
	features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25, random_state=42)
	print('[!info] Datapoints = (Total: {0}) =  ((Test: {1}) + (Train: {2}))'.format(labels.shape[0], labels_test.shape[0], labels_train.shape[0]))
	print("----")
	print()

	print('[!info] Input feature vector dimensionality: {0}'.format(input_dim))
	print('[!info] Hidden units: {0}'.format(FLAGS.hidden_units))
	print('[!info] Input dataset number of values to be predicted: {0}'.format(num_predicted_values))
	print('[!info] Threshold : {0}'.format(FLAGS.threshold))
	print('[!info] mu : {0}'.format(FLAGS.mu))
	print('[!info] Maximum Iterations : {0}'.format(FLAGS.max_epoch))
	print("----")
	print()

	class Regressor(nn.Module):
		def __init__(self):
			super(Regressor, self).__init__()
			self.fc1 = nn.Linear(in_features=input_dim, out_features=FLAGS.hidden_units, bias=False)
			self.fc2 = nn.Linear(in_features=FLAGS.hidden_units, out_features=num_predicted_values, bias=False)


		def forward(self, z):
			y = F.relu(self.fc1(z))
			x = self.fc2(y)
			return x

	def relu(inp):
	  return inp.clip(min=0)

	def forward_pass(W1, W2, inp):
	  output = relu(W2*relu(W1*inp))
	  return output.T

	model = Regressor()

	W1, W2 = np.matrix(list(model.parameters())[0].data.numpy()), np.matrix(list(model.parameters())[1].data.numpy())

	features_train_var, features_test_var, labels_train_var, labels_test_var = np.matrix(features_train).T, np.matrix(features_test).T, np.matrix(labels_train).T, np.matrix(labels_test).T

	with open(log_file, 'w') as log:
		log.write('Epoch\tLoss\n')

	start_time = timeit.default_timer()

	losses = []
	mu = FLAGS.mu
	prev_loss = 0.
	H = relu(np.matrix(W1*features_train_var))

	# Actually runs for max_epoch number of times. Since loss in the first iteration is just the result of the random wt. matrix.
	for epoch in range(0, FLAGS.max_epoch):
		loss = np.power(np.linalg.norm(labels_train_var - forward_pass(W1, W2, features_train_var).T), 2)
		losses.append(loss)
		if(abs(loss - prev_loss) < FLAGS.threshold):
			print('[Results] Threshold value reached at iteration: {0}'.format(epoch))
			break
		prev_loss = loss
	  
		# Matrix factorization step
		W1 = H*np.linalg.pinv(features_train_var)
		W2 = labels_train_var*np.linalg.pinv(H)
		H = np.linalg.inv(W2.T*W2 + mu*np.identity(FLAGS.hidden_units)) * (W2.T*labels_train_var + mu*W1*features_train_var)
		H = H.clip(min = 0)
	  
		with open(log_file, 'a') as log:
			log.write('{0}\t{1}\n'.format(
				epoch,
				loss        
			))

		if (epoch + 1) % 5 == 0 or (epoch + 1) == FLAGS.max_epoch:    
			print("Loss for epoch {0} : {1} ".format(epoch, loss))
			print('')

	stop_time = timeit.default_timer()
	total_time = stop_time - start_time

	print("[Results] Total time in seconds: {0}".format(total_time))

	losses = np.array(losses)
	plt.plot(losses)
	plt.xlabel('Epoch')
	plt.ylabel('Loss ')
	plt.title('{0}'.format(FLAGS.dataset.replace('-', ' ').capitalize()))
	plt.locator_params(axis='y', nbins=12)

	plt.savefig(plot_save)
	
	## Calculate the Train Data RMSE and MAE.
	predictions = np.array(forward_pass(W1, W2, features_train_var))
	mae = 0.
	rmse = 0.
	mae = mean_absolute_error(labels_train, predictions)
	mse = mean_squared_error(labels_train, predictions)
	rmse = pow(mse, 0.5)
	print("[Results] Train Data Metrics = mae : {0}, rmse : {1}".format(mae, rmse))


	## Calculate the Test Data RMSE and MAE
	predictions = np.array(forward_pass(W1, W2, features_test_var))

	mae = 0.
	rmse = 0.
	mae = mean_absolute_error(labels_test, predictions)
	mse = mean_squared_error(labels_test, predictions)
	rmse = pow(mse, 0.5)
	print("[Results] Test Data Metrics = mae : {0}, rmse : {1}".format(mae, rmse))

print(" ")
print("======================================================")