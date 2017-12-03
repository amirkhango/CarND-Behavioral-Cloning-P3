from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPooling2D, Cropping2D
from keras.regularizers import l2
def myCNN(resized_shape=64):
	model = Sequential()
	#model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
	model.add(Cropping2D(input_shape=(resized_shape,resized_shape,3)))
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
def nvidia_model():
  input_shape = (64,64,3)
  model = Sequential()
  model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
  model.add(Conv2D(24, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Conv2D(36, 5, 5, border_mode='valid', subsample =(2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Conv2D(48, 5, 5, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Conv2D(64, 3, 3, border_mode='same', subsample = (2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Conv2D(64, 3, 3, border_mode='valid', subsample = (2,2), W_regularizer = l2(0.001)))
  model.add(Activation('relu'))
  model.add(Flatten())
  model.add(Dense(80, W_regularizer = l2(0.001)))
  model.add(Dropout(0.5))
  model.add(Dense(40, W_regularizer = l2(0.001)))
  model.add(Dropout(0.5))
  model.add(Dense(16, W_regularizer = l2(0.001)))
  model.add(Dropout(0.5))
  model.add(Dense(10, W_regularizer = l2(0.001)))
  model.add(Dense(1, W_regularizer = l2(0.001)))
  return model
