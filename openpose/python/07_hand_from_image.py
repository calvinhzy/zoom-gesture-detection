# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
from sys import platform
import argparse
import time
import numpy as np
import pickle
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import *
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data
from joblib import dump, load

model_name = 'model.sav'

gestures = ['thumbs_up', 'thumbs_down', 'pause', 'cross', 'ok_sign', 'raise_hand', 'no_gesture']
num_classes = len(gestures)

confidence_threshold = 0.3
num_points = 42
feature_per_point = 2
percent_points_confident = 0.4


def format_input(X):
	formatted_X = np.concatenate((X[0].flatten(), X[1].flatten()))
	# print(formatted_X.shape)
	# print(formatted_X)
	formatted_X = formatted_X.reshape((1,-1))
	return formatted_X

def preprocess(X):
	processed_X = []
	for i, x in enumerate(X):
		confidences = x[2::3]
		if np.count_nonzero(confidences > confidence_threshold) > percent_points_confident * num_points:
			# processed_X.append(x)
			processed_X.append([a for j,a in enumerate(x) if (j-2)%3!=0])
	processed_X = np.array(processed_X)
	return processed_X

def feature_selection(X):
	processed_X = []
	for i, x in enumerate(X):

		# curr = []
		curr = [a for a in x]
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

		curr.append((x[42 + 16] - x[42 + 10]) ** 2 + (x[42 + 17] - x[42 + 11]) ** 2)
		curr.append((x[42 + 24] - x[42 + 18]) ** 2 + (x[42 + 25] - x[42 + 19]) ** 2)
		curr.append((x[42 + 32] - x[42 + 26]) ** 2 + (x[42 + 33] - x[42 + 27]) ** 2)
		curr.append((x[42 + 40] - x[42 + 34]) ** 2 + (x[42 + 41] - x[42 + 35]) ** 2)

		# distance btw thumb with index finger
		curr.append((x[16] - x[8]) ** 2 + (x[17] - x[9]) ** 2)
		curr.append((x[14] - x[6]) ** 2 + (x[15] - x[7]) ** 2)

		curr.append((x[42 + 16] - x[42 + 8]) ** 2 + (x[42 + 17] - x[42 + 9]) ** 2)
		curr.append((x[42 + 14] - x[42 + 6]) ** 2 + (x[42 + 15] - x[42 + 7]) ** 2)

		# highest point
		curr.append(np.argmax(x[1::2]))
		# lowest point
		curr.append(np.argmin(x[1::2]))


		# curr = x
		processed_X.append(curr)
	processed_X = np.array(processed_X)
	return processed_X

loaded_clf = load(model_name)
def predict(X):
    formatted_X = format_input(X)
    processed_X = preprocess(formatted_X)
    processed_X = feature_selection(processed_X)
    print(processed_X.shape)
    if (len(processed_X)==0):
	    print('no_gesture')
    else:
      pred = loaded_clf.predict(processed_X)
      print(gestures[int(pred)])


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

model_path = 'model.pt'
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
model = Model()
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.double()
model.to(device)

def predictWithDeepLearning(X):
	formatted_X = format_input(X)
	processed_X = preprocess(formatted_X)
	if (len(processed_X) == 0):
		print('no_gesture')
	else:
		processed_X = torch.from_numpy(processed_X).double()
		processed_X = processed_X.to(device)
		y_pred = model(processed_X)
		_, predicted = torch.max(y_pred.data, 1)
		# print(predicted)
		print(gestures[int(predicted)])



try:
	# Import Openpose (Windows/Ubuntu/OSX)
	dir_path = os.path.dirname(os.path.realpath(__file__))
	try:
		# Change these variables to point to the correct folder (Release/x64 etc.)
		sys.path.append(dir_path + '/../bin/python/openpose/Release');
		os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../x64/Release;' + dir_path + '/../bin;'
		import pyopenpose as op
	except ImportError as e:
		print(
			'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
		raise e

	# Flags
	parser = argparse.ArgumentParser()
	parser.add_argument("--image_path", default="../examples/media/COCO_val2014_000000000241.jpg",
	                    help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
	parser.add_argument("--net_resolution -1x384", default="../examples/media/COCO_val2014_000000000241.jpg",
						help="Process an image. Read all standard formats (jpg, png, bmp, etc.).")
	args = parser.parse_known_args()

	# Custom Params (refer to include/openpose/flags.hpp for more parameters)
	params = dict()
	params["model_folder"] = "../models/"
	params["hand"] = True
	params["hand_detector"] = 2
	params["body"] = 0

	# Add others in path?
	for i in range(0, len(args[1])):
		curr_item = args[1][i]
		if i != len(args[1]) - 1:
			next_item = args[1][i + 1]
		else:
			next_item = "1"
		if "--" in curr_item and "--" in next_item:
			key = curr_item.replace('-', '')
			if key not in params:  params[key] = "1"
		elif "--" in curr_item and "--" not in next_item:
			key = curr_item.replace('-', '')
			if key not in params: params[key] = next_item

	# Construct it from system arguments
	# op.init_argv(args[1])
	# oppython = op.OpenposePython()

	# Starting OpenPose
	opWrapper = op.WrapperPython()
	opWrapper.configure(params)
	opWrapper.start()

	# Read image and face rectangle locations
	imageToProcess = cv2.imread(args[0].image_path)
	print(imageToProcess.shape)
	handRectangles = [
		# Left/Right hands person 0
		[op.Rectangle(160, 160, 480, 480), op.Rectangle(0., 160., 480, 480)]
		# [op.Rectangle(0, 0, 480, 480), op.Rectangle(160., 0, 480, 480)]
		# [op.Rectangle(160, 0, 480, 480), op.Rectangle(0, 0, 480, 480)]
		# [
		#     op.Rectangle(320.035889, 377.675049, 69.300949, 69.300949),
		#     op.Rectangle(0., 0., 0., 0.),
		# ],
		# # Left/Right hands person 1
		# [
		#     op.Rectangle(80.155792, 407.673492, 80.812706, 80.812706),
		#     op.Rectangle(46.449715, 404.559753, 98.898178, 98.898178),
		# ],
		# # Left/Right hands person 2
		# [
		#     op.Rectangle(185.692673, 303.112244, 157.587555, 157.587555),
		#     op.Rectangle(88.984360, 268.866547, 117.818230, 117.818230),
		# ]
	]

	import cv2

	cam = cv2.VideoCapture(0)

	cv2.namedWindow("test")

	img_counter = 0

	data = []
	capturing = False

	while True:
		ret, frame = cam.read()
		if not ret:
			print("failed to grab frame")
			break
		cv2.imshow("test", frame)

		k = cv2.waitKey(1)
		if k % 256 == 27:
			# ESC pressed
			print("Escape hit, closing...")
			break
		elif k % 256 == 32:
			capturing = True
			# SPACE pressed
			#         # Create new datum
		# if capturing:
			datum = op.Datum()
			datum.cvInputData = frame
			datum.handRectangles = handRectangles

			# Process and display image
			opWrapper.emplaceAndPop(op.VectorDatum([datum]))

			data.extend(datum.handKeypoints[0])
			data.extend(datum.handKeypoints[1])
			# print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
			# print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))

			predict(datum.handKeypoints)
			# predictWithDeepLearning(datum.handKeypoints)

			cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)

	# output_file = 'ok_sign.csv'
	# print(len(data))
	# with open(output_file,'w') as output:
	# 	k = 0
	# 	for picture in data:
	# 		for i in range(21):
	# 			for j in range(3):
	# 				output.write(str(picture[i][j]))
	# 				output.write(',')
	# 		k += 1
	# 		if k==2:
	# 			output.write('\n')
	# 			k=0

	cam.release()

	cv2.destroyAllWindows()


except Exception as e:
	print(e)
	sys.exit(-1)
