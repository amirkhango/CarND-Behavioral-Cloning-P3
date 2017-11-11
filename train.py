import csv
import cv2
import numpy as np
import os
from keras.callbacks import EarlyStopping, ModelCheckpoint
from models import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pickle

np.random.seed(1337)  # for reproducibility
path_model='./MODEL'
path_log = './log'
nb_epoch=5
batch_size=32
hyperparams_name = 'myCNN'

def visual_log(history,save_path):
	# summarize history for loss
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	#plt.show()

	save_file = os.path.join(path_log, '{}.loss.png'.format(hyperparams_name))
	plt.savefig(save_file)
	print('Training loss visualizaiton has been saved in :', save_file)


def load_data():
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
	#X=np.array(images)[:30]
	#y=np.array(measurements)[:30]
        
	X=np.array(images)
	y=np.array(measurements)
	np.random.shuffle(X)
	np.random.shuffle(y)
	return X, y

def build_model():
	model = myCNN()
	model.compile(loss='mse', optimizer='adam')
	model.summary()
	return model

def main():

	print("loading data...")
	X_train, y_train = load_data()

	fname_param = os.path.join(path_model, '{}.best.h5'.format(hyperparams_name))
	
	early_stopping = EarlyStopping(monitor='val_mse', patience =5, mode='min')
	model_checkpoint = ModelCheckpoint(
	    fname_param, monitor='val_mse', verbose=0, save_best_only=True, mode='min')

	model = build_model()
	history = model.fit(X_train, y_train, validation_split=0.2,
	    shuffle=True,
	    nb_epoch=nb_epoch,
	    batch_size=batch_size,
	    callbacks=[early_stopping, model_checkpoint],)

	model.save_weights(os.path.join(path_model, '{}.h5'.format(hyperparams_name)), overwrite=True)
	pickle.dump((history.history), open(os.path.join(path_log, '{}.history.pkl'.format(hyperparams_name)), 'wb'))
	print('=' * 10)
	# list all data in history
	print(history.history.keys())

	visual_log(history=history.history, save_path=path_log)


if __name__ == '__main__':
	main()
	#logfile = os.path.join(path_log, '{}.history.pkl'.format(hyperparams_name))
	#print(logfile)
	#history = pickle.load(open(logfile, "rb"))
	#visual_log(history=history, save_path=path_log)
