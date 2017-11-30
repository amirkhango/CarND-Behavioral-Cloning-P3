import csv
import cv2
from PIL import Image
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
batch_size=2
hyperparams_name = 'myCNN'
save_file = os.path.join(path_log, '{}.loss.png'.format(hyperparams_name))

def visual_log(history,save_path):
	# summarize history for loss
	plt.plot(history['loss'])
	plt.plot(history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'val'], loc='upper left')
	#plt.show()
	
	plt.savefig(save_file)
	print('Training loss visualizaiton has been saved in :', save_file)

def load_samples():
	samples=[]
	with open('./data/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:
 			samples.append(line)

	return samples

def gen_data(samples, batch_size):
	
	#offset = 0 
	#print('number of samples is:', len(lines))	
	num_samples = len(samples)
	while True:
		for offset in range(0, num_samples, batch_size):
			
			images=[]
			measurements=[]
			batch_lines = samples[ offset : offset + batch_size]

			for line in batch_lines:

				source_path=line[0]
				filename = source_path.split('/')[-1]
				current_path='./data/IMG/'+ filename
				#current_path='./data/'+source_path
				image = np.asarray(Image.open(current_path))
				images.append(image)

				measurement = float(line[3])
				measurements.append(measurement)

		#X=np.array(images)[:30]
		#y=np.array(measurements)[:30]
	        
				X=np.array(images)
				y=np.array(measurements)
				np.random.shuffle(X)
				np.random.shuffle(y)

			yield X, y

def build_model():
	model = myCNN()
	model.compile(loss='mse', optimizer='adam')
	model.summary()
	return model

def main():

	print("loading data...")

	samples = load_samples()[:20]
	len_samples = len(samples)

	train_samples = samples[ :-int(len_samples*0.1) ]
	valid_samples = samples[ -int(len_samples*0.1): ]
	train_generator = gen_data(train_samples, batch_size)
	valid_generator = gen_data(valid_samples, batch_size)

	fname_param = os.path.join(path_model, '{}.best.h5'.format(hyperparams_name))
	
	early_stopping = EarlyStopping(monitor='val_mse', patience =5, mode='min')
	model_checkpoint = ModelCheckpoint(
	    fname_param, monitor='val_mse', verbose=0, save_best_only=True, mode='min')

	model = build_model()

	history = model.fit_generator(train_generator, 
		steps_per_epoch = len(train_samples) / batch_size, 
		validation_data = valid_generator,
		validation_steps = len(valid_samples) / batch_size,
	    nb_epoch = nb_epoch,
	    verbose = 0,
	    callbacks = [early_stopping, model_checkpoint])

	model.save(os.path.join(path_model, '{}.h5'.format(hyperparams_name)), overwrite=True)
	print('{} is saved in {}'.format('{}.h5'.format(hyperparams_name), path_model))
	pickle.dump((history.history), open(os.path.join(path_log, '{}.history.pkl'.format(hyperparams_name)), 'wb'))


	print('=' * 10)
	# list all data in history
	print(history.history.keys())

	visual_log(history=history.history, save_path=save_file)


if __name__ == '__main__':
	main()
	#logfile = os.path.join(path_log, '{}.history.pkl'.format(hyperparams_name))
	#print(logfile)
	#history = pickle.load(open(logfile, "rb"))
	#visual_log(history=history, save_path=path_log)
