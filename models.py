from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, Cropping2D

def myCNN():
	model = Sequential()
	model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
	model.add(Lambda(lambda x: x/255.0 - 0.5))
	model.add(Conv2D(16, kernel_size=(3, 3),
			 activation='relu',
			 )
	)
	model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
	model.add(BatchNormalization())	
	model.add(Conv2D(16, kernel_size=(3, 3),
                         activation='relu',
                         ))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(BatchNormalization())
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
	#model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(30, activation='relu'))
	#model.add(Dropout(0.5))
	model.add(Dense(1))

	return model
