import csv
import cv2
import numpy as np


lines=[]

with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images=[]
measurements=[]

for line in lines:
	source_path=line[0]
	filename = source_path.split('/')[-1]
	current_path='./data/IMG/'+ filename
	image = cv2.imread(current_path)
	images.append(image)

	measurement = float(line[3])
	measurements.append(measurement)

X_train=np.array(images)[:1002]
y_train=np.array(measurements)[:1002]

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 ))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2,
	shuffle=True, nb_epoch=2)

model.save('model.h5')


