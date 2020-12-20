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
feature_num = 183
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
			curr = []
			for j in range(0,len(x),3):
				if x[j+2]>confidence_threshold:
					curr.append(x[j])
					curr.append(x[j+1])
				else:
					curr.append(0)
					curr.append(0)
			# processed_X.append([a for j,a in enumerate(x) if (j-2)%3!=0 ])
			processed_X.append(curr)
			processed_Y.append(Y[i])
			continue
		confidences = x[2::3]
		if np.count_nonzero(confidences > confidence_threshold) > percent_points_confident * num_points:
			# print(np.array([a for j,a in enumerate(x) if (j-2)%3!=0]).shape)
			# print(np.array(x).shape)
			curr = []
			for j in range(0, len(x), 3):
				if x[j + 2] > confidence_threshold:
					curr.append(x[j])
					curr.append(x[j + 1])
				else:
					curr.append(0)
					curr.append(0)
			# processed_X.append([a for j,a in enumerate(x) if (j-2)%3!=0 ])
			processed_X.append(curr)
			# processed_X.append(x)
			processed_Y.append(Y[i])
	processed_X = np.array(processed_X)
	processed_Y = np.array(processed_Y)
	return processed_X, processed_Y

def feature_selection(X,Y):
	processed_X, processed_Y = [], []
	for i, x in enumerate(X):

		curr = [a for a in x]
		# curr = []
		# get each edge segment vector:
		for ii in range(2):
			for j in range(5):
				for k in range(1,5):
					if k == 1:
						index_x_1 = (ii * 21 + j * 4 + k) * 2
						index_x_2 = (ii * 21) * 2
						index_y_1 = (ii * 21 + j * 4 + k) * 2 + 1
						index_y_2 = (ii * 21) * 2 + 1
						# print(index_x_1, index_x_2)
						# print(index_y_1,index_y_2)
						# print(len(x))
						curr.append(x[index_x_1] - x[index_x_2])  # x
						curr.append(x[index_y_1] - x[index_y_2])  # y
					else:
						index_x_1 = (ii * 21 + j * 4 + k) * 2
						index_x_2 = (ii * 21 + j * 4 + k - 1) * 2
						index_y_1 = (ii * 21 + j * 4 + k) * 2 + 1
						index_y_2 = (ii * 21 + j * 4 + k - 1) * 2 + 1
						# print(index_x_1, index_x_2)
						# print(index_y_1,index_y_2)
						curr.append(x[index_x_1] - x[index_x_2])  # x
						curr.append(x[index_y_1] - x[index_y_2])  # y

		# manual feature selection
		# distance btw finger tips to palm
		curr.append((x[16] - x[10]) ** 2 + (x[17] - x[11]) ** 2)
		curr.append((x[24] - x[18]) ** 2 + (x[25] - x[19]) ** 2)
		curr.append((x[32] - x[26]) ** 2 + (x[33] - x[27]) ** 2)
		curr.append((x[40] - x[34]) ** 2 + (x[41] - x[35]) ** 2)

		curr.append((x[42+16] - x[42+10]) ** 2 + (x[42+17] - x[42+11]) ** 2)
		curr.append((x[42+24] - x[42+18]) ** 2 + (x[42+25] - x[42+19]) ** 2)
		curr.append((x[42+32] - x[42+26]) ** 2 + (x[42+33] - x[42+27]) ** 2)
		curr.append((x[42+40] - x[42+34]) ** 2 + (x[42+41] - x[42+35]) ** 2)

		# distance btw thumb with index finger
		curr.append((x[16]-x[8])**2+(x[17] - x[9])**2)
		curr.append((x[14] - x[6]) ** 2 + (x[15] - x[7]) ** 2)

		curr.append((x[42+16] - x[42+8]) ** 2 + (x[42+17] - x[42+9]) ** 2)
		curr.append((x[42+14] - x[42+6]) ** 2 + (x[42+15] - x[42+7]) ** 2)

		# distance between two hands
		curr.append((x[42 + 8] - x[8]) ** 2 + (x[42 + 9] - x[9]) ** 2)
		curr.append((x[42 + 16] - x[16]) ** 2 + (x[42 + 17] - x[17]) ** 2)
		curr.append((x[42 + 24] - x[24]) ** 2 + (x[42 + 25] - x[25]) ** 2)
		curr.append((x[42 + 32] - x[32]) ** 2 + (x[42 + 33] - x[33]) ** 2)
		curr.append((x[42 + 40] - x[40]) ** 2 + (x[42 + 41] - x[41]) ** 2)

		# highest point
		curr.append(np.argmax(x[1::2]))
		# lowest point
		curr.append(np.argmin(x[1::2]))


		# curr = x
		processed_X.append(curr)
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
	X_array, Y_array = feature_selection(X_array, Y_array)
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
			self.linear1 = nn.Linear(feature_num, 300)
			self.linear2 = nn.Linear(300, 100)
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


