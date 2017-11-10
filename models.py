from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D

def myCNN():
	model = Sequential()
	model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
	model.add(Conv2D(32, kernel_size=(3, 3),
			 activation='relu',
			 ))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	#model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(512, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(1))

        return model
