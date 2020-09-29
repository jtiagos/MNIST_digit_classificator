# SALVANDO MODELO PARA USO POSTERIOR
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
 
# load de: train/test dataset
def load_dataset():
	# load: dataset
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
 
#Modelo CNN
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
	# compilando modelo
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model
 
# "test harness" para validação do modelo
def run_test_harness():
	# load: dataset
	trainX, trainY, testX, testY = load_dataset()
	# Preparo dos pixels
	trainX, testX = prep_pixels(trainX, testX)
	# Definindo modelo
	model = define_model()
	# fit do modelo
	model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
	# salvando modelo final para uso posterior
	model.save('final_model.h5')
 
# MAIN, rodar "test harness"
run_test_harness()