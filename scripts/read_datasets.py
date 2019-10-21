import sklearn.datasets
from sklearn.model_selection import train_test_split
import sys
import numpy as np

def get_classification_dataset_dictionary(file_path, replacer, class_index, remove_value, remove_index = -1, split_character = ',', float_val = False):
	dataset = []
	with open(file_path, 'r') as file:
		for x in file:
			datapointRaw = list(x.replace('\n', '').split(split_character))
			for i in range(datapointRaw.count('')):
				datapointRaw.remove('')
			if remove_value:
				datapointRaw.pop(remove_index)
			datapoint = []
			for x in datapointRaw:
				val = replacer.get(x, 'ERROR')
				if val == 'ERROR':
					if(not float_val):
						val = int(x)
					else:
						val = float(x)
				datapoint.append(val)
			## swapping the class index and the last index items.
			if class_index != -1:
				class_value = datapoint[class_index]
				datapoint[class_index] = datapoint[-1]
				datapoint[-1] = class_value
			dataset.append(datapoint)
	return dataset

class read_classification_datasets():
	def __init__(self, dataset_name):
		loadfunction_name = 'load_{0}'.format(dataset_name)
		try:
			dataset = getattr(sklearn.datasets, loadfunction_name)()
			features, targets = dataset.data, dataset.target
		except AttributeError:
			print("[!info] Checking the dataset directory for the required dataset - {0}.".format(dataset_name))
			features, targets = read_classification_datasets.load_dataset(dataset_name)
		print("[!info] Successfully loaded the regression dataset \"{0}\"".format(dataset_name))
		self.features, self.targets = features, targets

	def load_dataset(dataset_name):
		if dataset_name == 'tic-tac-toe':
			replacer = {
				'o': 0,
				'x': 1,
				'b': 2,
				'positive': 1,
				'negative': 0
			}

			ds = get_classification_dataset_dictionary('../datasets/classification-datasets/tic-tac-toe.data', replacer, -1, False)
			ds = np.array(ds)

			features, targets = np.array(ds[:, 0:-1]), np.array(ds[:, -1])

		elif dataset_name == 'soybean-small':
			replacer = {
				'D1': 0,
				'D2': 1,
				'D3': 2,
				'D4': 3
			}

			ds = get_classification_dataset_dictionary('../datasets/classification-datasets/soybean-small.data', replacer, -1, False)

			ds = np.array(ds)
			features, targets = np.array(ds[:, 0:-1]), np.array(ds[:, -1])

		elif dataset_name == 'shuttle':
			replacer = {
				'1': 0,
				'2': 1,
				'3': 2,
				'4': 3,
				'*': 5
			}

			ds = get_classification_dataset_dictionary('../datasets/classification-datasets/shuttle-landing-control.data', replacer, 0, False)

			ds = np.array(ds)
			features, targets = np.array(ds[:, 0:-1]), np.array(ds[:, -1])

		elif dataset_name == 'glass':
			replacer = {
				'1': 0,
				'2': 1,
				'3': 2,
				'4': 3,
				'5': 4,
				'6': 5,
				'7': 6,
			}

			ds = get_classification_dataset_dictionary('../datasets/classification-datasets/glass.data', replacer, -1, True, 0, ',', True)

			ds = np.array(ds)
			features, targets = np.array(ds[:, 0:-1]), np.array(ds[:, -1])
			targets = np.array(targets, dtype=int)

		elif dataset_name == 'seeds':
			replacer = {
				'1': 0,
				'2': 1,
				'3': 2,
			}
	
			ds = get_classification_dataset_dictionary('../datasets/classification-datasets/seeds_dataset.txt', replacer, -1, False, -1, '\t', True)

			ds = np.array(ds)
			features, targets = np.array(ds[:, 0:-1]), np.array(ds[:, -1])
			targets = np.array(targets, dtype=int)

		elif dataset_name == 'fertility':
			replacer = {
				'N': 0,
				'O': 1,
			}
	
			ds = get_classification_dataset_dictionary('../datasets/classification-datasets/fertility_Diagnosis.txt', replacer, -1, False, -1, ',', True)

			ds = np.array(ds)
			features, targets = np.array(ds[:, 0:-1]), np.array(ds[:, -1])
			targets = np.array(targets, dtype=int)
		else:
			print("[!ERROR] Requested dataset doesn't exist. Please ensure that the corresponding file exists in the dataset directory.")
			exit(1)
		return features, targets

def get_regression_dataset_dictionary(file_path, replacer, class_index, remove_value, remove_index = -1, split_character = ',', float_val = False, skip_row = False):
	dataset = []
	with open(file_path, 'r') as file:
		for x in file:
			if skip_row:
				skip_row = False
				continue
			datapointRaw = list(x.replace('\n', '').replace('%', '').split(split_character))
			for i in range(datapointRaw.count('')):
				datapointRaw.remove('')
			if remove_value:
				datapointRaw.pop(remove_index)
			datapoint = []
			for x in datapointRaw:
				val = replacer.get(x, 'ERROR')
				if val == 'ERROR':
					if(not float_val):
						val = int(x)
					else:
						val = float(x)
				datapoint.append(val)
			## swapping the target index and the last index items.
			if class_index != -1 and type(class_index) == int:
				class_value = datapoint[class_index]
				datapoint[class_index] = datapoint[-1]
				datapoint[-1] = class_value
			dataset.append(datapoint)
	return dataset

class read_regression_datasets():
	def __init__(self, dataset_name):
		loadfunction_name = 'load_{0}'.format(dataset_name)
		try:
			dataset = getattr(sklearn.datasets, loadfunction_name)()
			features, targets = dataset.data, dataset.target
		except AttributeError:
			print("[!info] Checking the dataset directory for the required dataset - {0}.".format(dataset_name))
			features, targets = read_regression_datasets.load_dataset(dataset_name)
		print("[!info] Successfully loaded the regression dataset \"{0}\"".format(dataset_name))
		self.features, self.targets = features, targets

	def load_dataset(dataset_name):
		if dataset_name == 'o-ring':
			replacer = {
			}	
		
			ds = get_regression_dataset_dictionary('../datasets/regression-datasets/o-ring-erosion-only.data', replacer, 1, True, split_character=None)
			ds = np.array(ds)

			features, targets = np.array(ds[:, 0:-1]), np.array(ds[:, -1])

		elif dataset_name == 'real-estate-valuation':
			replacer = {
			}	
		
			ds = get_regression_dataset_dictionary('../datasets/regression-datasets/real-estate-valuation.data', replacer, -1, True, remove_index = 0, float_val = True, skip_row = True)
			ds = np.array(ds)

			features, targets = np.array(ds[:, 0:-1]), np.array(ds[:, -1])

		elif dataset_name == 'energy-efficiency':
			replacer = {
			}	
		
			ds = get_regression_dataset_dictionary('../datasets/regression-datasets/energy-efficiency-data.data', replacer, slice(-2, None, None), False, float_val = True, skip_row = True)
			ds = np.array(ds)

			features, targets = np.array(ds[:, 0:-2]), np.array(ds[:, -2:])


		elif dataset_name == 'stock-portfolio-performance':
			replacer = {
			}	
		
			ds = get_regression_dataset_dictionary('../datasets/regression-datasets/stock-portfolio-performance.data', replacer, slice(-6, None, None), True, remove_index = 0, float_val = True, skip_row = True)
			ds = np.array(ds)

			features, targets = np.array(ds[:, 0:6]), np.array(ds[:, -6:])

		elif dataset_name == 'concrete-slump':
			replacer = {
			}	
		
			ds = get_regression_dataset_dictionary('../datasets/regression-datasets/slump-test.data', replacer, slice(-3, None, None), True, remove_index = 0, float_val = True, skip_row = True)
			ds = np.array(ds)

			features, targets = np.array(ds[:, 0:-3]), np.array(ds[:, -3:])

		elif dataset_name == 'daily-demand-forecasting':
			replacer = {
			}	
		
			ds = get_regression_dataset_dictionary('../datasets/regression-datasets/daily-demand-forecasting-orders.data', replacer, -1, False, float_val = True, skip_row = True, split_character=';')
			ds = np.array(ds)

			features, targets = np.array(ds[:, 0:-1]), np.array(ds[:, -1])

		elif dataset_name == 'concrete-compressive-strength':
			replacer = {
			}	
		
			ds = get_regression_dataset_dictionary('../datasets/regression-datasets/concrete-compressive-strength.data', replacer, -1, False, float_val = True, skip_row = True)
			ds = np.array(ds)

			features, targets = np.array(ds[:, 0:-1]), np.array(ds[:, -1])

		elif dataset_name == 'airfoil-self-noise':
			replacer = {
			}	
		
			ds = get_regression_dataset_dictionary('../datasets/regression-datasets/airfoil-self-noise.data', replacer, -1, False, float_val = True, split_character='\t')
			ds = np.array(ds)

			features, targets = np.array(ds[:, 0:-1]), np.array(ds[:, -1])

		else:
			print("[!ERROR] Requested dataset doesn't exist. Please ensure that the corresponding file exists in the dataset directory.")
			exit(1)
		return features, targets

if __name__ == "__main__":
	classification_problems = ['iris', 'digits', 'wine', 'breast_cancer', 'tic-tac-toe', 'soybean-small', 'shuttle', 'glass', 'seeds', 'fertility']
	regression_problems = ['boston', 'linnerud', 'real-estate-valuation', 'energy-efficiency', 'stock-portfolio-performance', 'concrete-slump', 'daily-demand-forecasting', 'concrete-compressive-strength', 'airfoil-self-noise', 'o-ring']

	for dataset_name in classification_problems:
		loaded_dataset = read_classification_datasets(dataset_name)
		print(len(loaded_dataset.features), len(loaded_dataset.targets))
		print()

	for dataset_name in regression_problems:
		loaded_dataset = read_regression_datasets(dataset_name)
		print(len(loaded_dataset.features), len(loaded_dataset.targets))
		print()