import numpy as np

def NN_classify(X_train, y_train, X_test, y_test):
	from keras.models import Sequential
	from keras.layers.core import Dense, Dropout, Activation
	from keras.optimizers import SGD

	model = Sequential()
	# Dense(64) is a fully-connected layer with 64 hidden units.
	# in the first layer, you must specify the expected input data shape:
	# here, 20-dimensional vectors.
	model.add(Dense(64, input_dim=13, init='glorot_uniform'))
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(64, init='uniform'))
	model.add(Activation('sigmoid'))
	model.add(Dropout(0.5))
	model.add(Dense(3, init='uniform'))
	model.add(Activation('softmax'))

	model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

	model.fit(X_train, y_train, nb_epoch=30, batch_size=1, show_accuracy=True)
	score = model.evaluate(X_test, y_test, batch_size=1, show_accuracy=True)
	print "loss: ", score[0]
	print "accuracy: ", score[1]


data = np.genfromtxt('winedata.txt', delimiter=',')
data = np.random.permutation(data)
data = np.array(data)
y_temp = data[:, 0]
X = data[:, 1:]

y = list()
for yi in y_temp:
	if yi == 1:
		y.append([1,0,0])
	elif yi == 2:
		y.append([0,1,0])
	elif yi == 3:
		y.append([0,0,1])

X_train = X[:150]
X_test = X[150:]
y_train = y[:150]
y_test = y[150:]

NN_classify(X_train, y_train, X_test, y_test)