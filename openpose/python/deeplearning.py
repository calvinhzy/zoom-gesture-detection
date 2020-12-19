import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.signal as signal
import numpy as np
import pickle
from joblib import dump, load
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.utils import data

confidence_threshold = 0.3
num_points = 42
feature_per_point = 2
percent_points_confident = 0.4


gestures = ['thumbs_up', 'thumbs_down', 'pause', 'cross', 'ok_sign', 'raise_hand', 'no_gesture']
num_classes = len(gestures)
model_name = 'model.sav'

def load_data(file_name, class_label, X, Y):
	with open(file_name) as input_file:
		line = input_file.readline()
		while line:
			X.append(line[:-1].split(',')[:-1])
			Y.append(class_label)
			line = input_file.readline()


def preprocess(X, Y):
	processed_X, processed_Y = [], []
	for i, x in enumerate(X):
		if gestures[int(Y[i])] == 'no_gesture':
			processed_X.append([a for j,a in enumerate(x) if (j-2)%3!=0])
			processed_Y.append(Y[i])
			continue
		confidences = x[2::3]
		if np.count_nonzero(confidences > confidence_threshold) > percent_points_confident * num_points:
			# print(np.array([a for j,a in enumerate(x) if (j-2)%3!=0]).shape)
			# print(np.array(x).shape)
			processed_X.append([a for j,a in enumerate(x) if (j-2)%3!=0])
			# processed_X.append(x)
			processed_Y.append(Y[i])
	processed_X = np.array(processed_X)
	processed_Y = np.array(processed_Y)
	return processed_X, processed_Y

def train():
	X = []
	Y = []

	for i, gesture in enumerate(gestures):
		load_data(gesture + '.csv', i, X, Y)
	X = np.array(X)
	Y = np.array(Y)
	X = X.astype(np.float64)
	print(X.shape)
	print(Y.shape)

	X_array, Y_array = preprocess(X, Y)
	# X_array, Y_array = np.array(X), np.array(Y)
	print(X_array.shape)
	print(Y_array.shape)


	from sklearn.model_selection import train_test_split
	train_x, test_x, train_y, test_y = train_test_split(X_array, Y_array,
	                                                    test_size=0.2,
	                                                    random_state=0,
	                                                    stratify=Y_array)
	train_x, val_x, train_y, val_y = train_test_split(train_x, train_y,
	                                                    test_size=0.15,
	                                                    random_state=0,
	                                                    stratify=train_y)
	print(train_x.shape, test_x.shape, val_x.shape, train_y.shape, test_y.shape, val_y.shape)


	cuda = torch.cuda.is_available()
	device = torch.device("cuda" if cuda else "cpu")

	# device = "cpu"

	class MyDataset(data.Dataset):
		def __init__(self, X, Y):
			self.X = X
			self.Y = Y

		def __len__(self):
			return len(self.Y)

		def __getitem__(self, index):
			return self.X[index], self.Y[index]

	train_dataset = MyDataset(train_x, train_y)
	train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=32)

	class Model(nn.Module):

		def __init__(self):
			super(Model, self).__init__()
			self.linear1 = nn.Linear(num_points*feature_per_point, 200)
			self.linear2 = nn.Linear(200, 100)
			self.linear3 = nn.Linear(100, num_classes)
			self.relu = torch.nn.ReLU()

		def forward(self, input_seq):
			x = self.linear1(input_seq)
			x = self.relu(x)
			x = self.linear2(x)
			x = self.relu(x)
			x = self.linear3(x)
			return x

	model = Model()
	model.to(device)
	model = model.double()
	loss_function = nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters())

	def test_model(test_loader, print_accuracy=False):
		correct = 0
		total = 0
		class_correct = list(0. for i in range(num_classes))
		class_total = list(0. for i in range(num_classes))
		with torch.no_grad():
			for seq, labels in test_loader:
				seq = seq.to(device)
				labels = labels.to(device)
				optimizer.zero_grad()

				y_pred = model(seq)
				_, predicted = torch.max(y_pred.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
				c = (predicted == labels).squeeze()
				label = labels[0]
				class_correct[label] += c.item()
				class_total[label] += 1

		accuracy = 100 * correct / total
		if print_accuracy:
			print('Accuracy of the network %d %%' % (
				accuracy))
		accuracy_per_class = []
		for i in range(num_classes):
			accuracy_per_class.append(100 * class_correct[i] / class_total[i])
			if print_accuracy:
				print('Accuracy of %5s : %2d %%' % (
					gestures[i], accuracy_per_class[i]))
		return accuracy, accuracy_per_class

	epochs = 50
	val_dataset = MyDataset(val_x, val_y)
	val_loader = data.DataLoader(val_dataset)

	print('initial accuracies')
	test_model(val_loader, True)

	val_accuracy = []
	val_accuracy_per_class = []
	for i in range(epochs):
		for seq, labels in train_loader:
			seq = seq.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()

			y_pred = model(seq)

			labels = labels.long()
			single_loss = loss_function(y_pred, labels)
			single_loss.backward()
			optimizer.step()

		print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
		accu, accu_per_class = test_model(val_loader)
		val_accuracy.append(accu)
		val_accuracy_per_class.append(accu_per_class)

	print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
	saved_model_path = 'model.pt'
	torch.save({
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict()
	}, saved_model_path)

	print(val_accuracy)
	print(val_accuracy_per_class)

	import matplotlib.pyplot as plt
	plt.plot(val_accuracy)
	plt.title('Validation accuracy')

	test_dataset = MyDataset(test_x, test_y)
	test_loader = data.DataLoader(test_dataset)

	correct = 0
	total = 0
	class_correct = list(0. for i in range(num_classes))
	class_total = list(0. for i in range(num_classes))
	with torch.no_grad():
		for seq, labels in test_loader:
			seq = seq.to(device)
			labels = labels.to(device)
			optimizer.zero_grad()

			y_pred = model(seq)
			_, predicted = torch.max(y_pred.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			c = (predicted == labels).squeeze()
			label = labels[0]
			class_correct[label] += c.item()
			class_total[label] += 1

	print('Accuracy of the network %d %%' % (
			100 * correct / total))
	for i in range(num_classes):
		print('Accuracy of %5s : %2d %%' % (
			gestures[i], 100 * class_correct[i] / class_total[i]))

train()


