import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import validation_curve
from sklearn.tree import DecisionTreeRegressor
import pickle
from joblib import dump, load
import matplotlib.pyplot as plt

confidence_threshold = 0.3
num_points = 42
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
			processed_X.append(x)
			processed_Y.append(Y[i])
			continue
		confidences = x[2::3]
		if np.count_nonzero(confidences > confidence_threshold) > percent_points_confident * num_points:
			processed_X.append(x)
			processed_Y.append(Y[i])
	processed_X = np.array(processed_X)
	processed_Y = np.array(processed_Y)
	return processed_X, processed_Y


def testPerformance(clf, X, Y, cv):
	# print(X.shape)
	# print(Y.shape)
	scores = []
	classIncorrect = [0] * num_classes
	classTotal = [0] * num_classes
	for train_index, test_index in cv.split(X):
		#     print("Train Index: ", train_index)
		#     print("Test Index: ", test_index)
		X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
		clf.fit(X_train, y_train)
		pred = clf.predict(X_test)
		for i, t in enumerate(y_test):
			if y_test[i] != pred[i]:
				#             print(y_test[i],pred[i])
				classIncorrect[y_test[i]] += 1
			classTotal[y_test[i]] += 1
		s = clf.score(X_test, y_test)
		scores.append(s)
	print('Num incorrect for each class: ', classIncorrect, 'Num total for each class: ', classTotal)
	classAccuracy = [(classTotal[i] - classIncorrect[i]) / classTotal[i] for i in range(len(classTotal))]
	print("Class accuracy")
	print(classAccuracy)
	print("Mean accuracy:" + str(np.mean(scores)))
	print(scores)


def plotTraining(clf, X_array, Y_array):
	np.set_printoptions(precision=2)

	# Plot non-normalized confusion matrix
	titles_options = [("Confusion matrix, without normalization", None),
	                  ("Normalized confusion matrix", 'true')]
	for title, normalize in titles_options:
		disp = plot_confusion_matrix(clf, X_array, Y_array,
		                             display_labels=gestures,
		                             cmap=plt.cm.Blues,
		                             normalize=normalize)
		disp.ax_.set_title(title)

		print(title)
		print(disp.confusion_matrix)

	plt.show()

def testKNN(cv, X, Y):
	bestK = -1
	bestScore = 0
	for i in range(1, 25):
		clf = KNeighborsClassifier(n_neighbors=i)
		local_score = 0
		for train_index, test_index in cv.split(X):
			X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
			clf.fit(X_train, y_train)
			local_score += clf.score(X_test, y_test)
		print(i, local_score/10)
		if local_score > bestScore:
			bestScore = local_score
			bestK = i
	print(bestK)

def testModelsWithCurve(cv,X,Y):
	# parameter_range = np.arange(1, 10, 1)
	# train_score, test_score = validation_curve(KNeighborsClassifier(), X, Y,
	#                                            param_name="n_neighbors",
	#                                            param_range=parameter_range,
	#                                            cv=cv, scoring="accuracy")



	parameter_range = np.arange(1, 15, 1)
	train_score, test_score = validation_curve(RandomForestClassifier(), X, Y,
	                                           param_name="max_depth",
	                                           param_range=parameter_range,
	                                           cv=cv, scoring="accuracy")

	# Calculating mean and standard deviation of training score
	mean_train_score = np.mean(train_score, axis=1)
	std_train_score = np.std(train_score, axis=1)

	# Calculating mean and standard deviation of testing score
	mean_test_score = np.mean(test_score, axis=1)
	std_test_score = np.std(test_score, axis=1)

	# Plot mean accuracy scores for training and testing scores
	plt.plot(parameter_range, mean_train_score,
	         label="Training Score", color='b')
	plt.plot(parameter_range, mean_test_score,
	         label="Cross Validation Score", color='g')

	# Creating the plot
	plt.title("Validation Curve with RF Classifier")
	plt.xlabel("Max depth")
	plt.ylabel("Accuracy")
	plt.tight_layout()
	plt.legend(loc='best')
	plt.show()

def testRandomForest(cv, X, Y):

	bestK = -1
	bestScore = 0
	for i in range(1, 25):
		clf = RandomForestClassifier(max_depth=i, random_state=0)
		local_score = 0
		for train_index, test_index in cv.split(X):
			X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
			clf.fit(X_train, y_train)
			local_score += clf.score(X_test, y_test)
		print(i, local_score/10)
		if local_score > bestScore:
			bestScore = local_score
			bestK = i
	print(bestK)

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

	cv = KFold(n_splits=10, shuffle=True, random_state=10)
	# testKNN(cv,X_array, Y_array)
	# testModelsWithCurve(cv,X_array, Y_array)
	# clf = KNeighborsClassifier(n_neighbors=2) # also 100% accuracy
	# clf = SVC(kernel='linear') # also 100% accuracy
	# testRandomForest(cv, X_array, Y_array)
	# clf = RandomForestClassifier(max_depth=7, random_state=0) # also 100% accuracy

	clf = MLPClassifier(solver='adam', alpha=1e-5,
	hidden_layer_sizes = (100), random_state = 1) # ok performance


	testPerformance(clf, X_array, Y_array, cv)
	plotTraining(clf, X_array, Y_array)


	# dump(clf, model_name)




train()
