# TREINANDO MODELO
#Imports necessários
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
 
# IMPORT do dataset
def load_dataset():
	# dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# Remodelagem dos dados para um único canal
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# valores alvo - codificação rápida
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY
 
# Escalonamento de pixels
def prep_pixels(train, test):
	# converção int to float
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalização: range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return: imagens normalizadas
	return train_norm, test_norm
 
# Modelo CNN
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile do modelo
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
 
# Avaliação do modelo usando k-fold - cross-validation
def evaluate_model(dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# preparação: cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
	# enumerar divisões
	for train_ix, test_ix in kfold.split(dataX):
		# definir modelo
		model = define_model()
		# seleção de linhas para treinar e testar
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit do modelo
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# avaliar modelo
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# print scores
		scores.append(acc)
		histories.append(history)
	return scores, histories
 
# plot: diagnóstico de aprendizado
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# PLot de Cross Entropy
		pyplot.subplot(2, 1, 1)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot de Accuracy
		pyplot.subplot(2, 1, 2)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
	pyplot.show()
 
# Sumarização
def summarize_performance(scores):
	# print: summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box e whisker plots dos resultados
	pyplot.boxplot(scores)
	pyplot.show()
 
# execute o test_harness para avaliar um modelo
def run_test_harness():
	# load do dataset
	trainX, trainY, testX, testY = load_dataset()
	# preparação do pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# Avaliando o modelo
	scores, histories = evaluate_model(trainX, trainY)
	# Curvas de aprendizado
	summarize_diagnostics(histories)
	# Sumarizando os estimadores
	summarize_performance(scores)
 
# MAIN - execute o "test harness"
run_test_harness()