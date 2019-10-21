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

parser.add_argument('--log_directory', type=str, default='../results/backpropagation/logs', help="destination directory to save the training logs")
parser.add_argument('--log_file', type=str, default='boston_backprop.log', help="text filename to save training logs")
parser.add_argument('--plot_directory', type=str, default='../results/backpropagation/plots', help="destination directory to save the loss vs iteration plots")
parser.add_argument('--plot_file', type=str, default='boston_backprop.png', help=".png filename to save the loss vs iteration plots")
parser.add_argument('--dataset', type=str, default='boston', help="dataset to run the script on")

parser.add_argument('--max_epoch', type=int, default=15000, help="flag to indicate the maximum epochs for training")
parser.add_argument('--initial_learning_rate', type=float, default=0.01, help="starting learning rate")
parser.add_argument('--threshold', type=float, default=1e-3, help="less than threshold value change in subsequent losses implies convergence")
parser.add_argument('--hidden_units', type=int, default=100, help="number of units in the hidden layer of the neural network")
parser.add_argument('--is_classification', type=bool, default=False, help="Boolean to indicate if the problem at hand is a classification problem.")


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

	features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25, random_state=42)

	print('[!info] Datapoints = (Total: {0}) =  ((Test: {1}) + (Train: {2}))'.format(len(labels), len(labels_test), len(labels_train)))
	print("----")
	print()

	print('[!info] Input feature vector dimensionality: {0}'.format(input_dim))
	print('[!info] Hidden units: {0}'.format(FLAGS.hidden_units))
	print('[!info] Input dataset classes: {0}'.format(num_classes))
	print('[!info] Threshold : {0}'.format(FLAGS.threshold))
	print('[!info] Initial learning rate : {0}'.format(FLAGS.initial_learning_rate))
	print('[!info] Maximum Iterations : {0}'.format(FLAGS.max_epoch))
	print("----")
	print()

	start_time = timeit.default_timer()

	features_train_var = Variable(torch.FloatTensor(features_train), requires_grad = False)
	labels_train_var = Variable(torch.FloatTensor(labels_train), requires_grad = False)
	features_test_var = Variable(torch.FloatTensor(features_test), requires_grad = False)
	labels_test_var = Variable(torch.FloatTensor(labels_test), requires_grad = False)
			  
	class Classifier(nn.Module):
		def __init__(self):
			super(Classifier, self).__init__()
			self.fc1 = nn.Linear(in_features=input_dim, out_features=FLAGS.hidden_units, bias=False)
			self.fc2 = nn.Linear(in_features=FLAGS.hidden_units, out_features=num_classes,bias=False)


		def forward(self, z):
			y = F.relu(self.fc1(z))
			x = self.fc2(y)
			return x

	def l2_loss(out, label):
		norm = torch.norm(label - out)
		return torch.pow(norm, 2)

	model = Classifier()
	optimizer = optim.RMSprop(model.parameters(), lr=FLAGS.initial_learning_rate)
		
	with open(log_file, 'w') as log:
		log.write('Epoch\tLoss\n')

	losses = []
	prev_loss = 0.

	for epoch in range(0, FLAGS.max_epoch):
		out = model(features_train_var)
		loss = l2_loss(labels_train_var, out)
		losses.append(loss.data)
	  
		if(abs(loss.data[0] - prev_loss) < FLAGS.threshold):
			print('[Results] Threshold value reached at iteration: {0}'.format(epoch))
			break
		prev_loss = loss.data[0]
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	  
		with open(log_file, 'a') as log:
			log.write('{0}\t{1}\n'.format(
				epoch,
				loss.data[0]        
			))

		if (epoch + 1) % 5 == 0 or (epoch + 1) == FLAGS.max_epoch:
			print("Loss for epoch {0} : {1} ".format(epoch, loss.data[0]))
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

	for num in range(len(features_train_var)):
		predictions.append(model(features_train_var[num]))
		
	score = 0
	for num in range(len(predictions)):
		if(np.argmax(labels_train[num]) == np.argmax(predictions[num].data.numpy())):
			score += 1
	accuracy = float(score)/float(len(predictions))
	print("[Results] Train Accuracy = {0}".format(accuracy * 100.))

	## Calculate the Test Data Accuracy
	predictions = []

	for num in range(len(features_test_var)):
		predictions.append(model(features_test_var[num]))
	score = 0
	for num in range(len(predictions)):
		if(np.argmax(labels_test[num]) == np.argmax(predictions[num].data.numpy())):
			score += 1
	accuracy = float(score)/float(len(predictions))
	print("[Results] Accuracy Test = {0}".format(accuracy * 100.))

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

	

	features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.25, random_state=42)
	print('[!info] Datapoints = (Total: {0}) =  ((Test: {1}) + (Train: {2}))'.format(labels.shape[0], labels_test.shape[0], labels_train.shape[0]))
	print("----")
	print()

	print('[!info] Input feature vector dimensionality: {0}'.format(input_dim))
	print('[!info] Hidden units: {0}'.format(FLAGS.hidden_units))
	print('[!info] Input dataset number of values to be predicted: {0}'.format(num_predicted_values))
	print('[!info] Threshold : {0}'.format(FLAGS.threshold))
	print('[!info] Initial learning rate : {0}'.format(FLAGS.initial_learning_rate))
	print('[!info] Maximum Iterations : {0}'.format(FLAGS.max_epoch))
	print("----")
	print()

	start_time = timeit.default_timer()

	features_train_var = Variable(torch.FloatTensor(features_train), requires_grad = False)
	labels_train_var = Variable(torch.FloatTensor(labels_train), requires_grad = False)
	features_test_var = Variable(torch.FloatTensor(features_test), requires_grad = False)
	labels_test_var = Variable(torch.FloatTensor(labels_test), requires_grad = False)
			  
	class Regressor(nn.Module):
		def __init__(self):
			super(Regressor, self).__init__()
			self.fc1 = nn.Linear(in_features=input_dim, out_features=FLAGS.hidden_units, bias=False)
			self.fc2 = nn.Linear(in_features=FLAGS.hidden_units, out_features=num_predicted_values,bias=False)


		def forward(self, z):
			y = F.relu(self.fc1(z))
			x = self.fc2(y)
			return x

	def l2_loss(out, label):
		norm = torch.norm(label - out)
		return torch.pow(norm, 2)

	model = Regressor()
	optimizer = optim.RMSprop(model.parameters(), lr=FLAGS.initial_learning_rate)

	with open(log_file, 'w') as log:
		log.write('Epoch\tLoss\n')

	losses = []
	prev_loss = 0.

	for epoch in range(0, FLAGS.max_epoch):
		out = model(features_train_var)
		loss = l2_loss(labels_train_var, out)
		losses.append(loss.data)
	  
		if(abs(loss.data[0] - prev_loss) < FLAGS.threshold):
			print('[Results] Threshold value reached at iteration: {0}'.format(epoch))
			break
		prev_loss = loss.data[0]
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	  
		with open(log_file, 'a') as log:
			log.write('{0}\t{1}\n'.format(
				epoch,
				loss.data[0]        
			))

		if (epoch + 1) % 5 == 0 or (epoch + 1) == FLAGS.max_epoch:
			print("Loss for epoch {0} : {1} ".format(epoch, loss.data[0]))
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
	predictions = model(features_train_var).data.numpy()
	mae = 0.
	rmse = 0.
	mae = mean_absolute_error(labels_train, predictions)
	mse = mean_squared_error(labels_train, predictions)
	rmse = pow(mse, 0.5)
	print("[Results] Train Data Metrics = mae : {0}, rmse : {1}".format(mae, rmse))


	## Calculate the Test Data RMSE and MAE
	predictions = model(features_test_var).data.numpy()

	mae = 0.
	rmse = 0.
	mae = mean_absolute_error(labels_test, predictions)
	mse = mean_squared_error(labels_test, predictions)
	rmse = pow(mse, 0.5)
	print("[Results] Test Data Metrics = mae : {0}, rmse : {1}".format(mae, rmse))

print(" ")
print("======================================================")