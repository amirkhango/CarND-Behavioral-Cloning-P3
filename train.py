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
from keras.models import load_model
import sklearn

np.random.seed(33)  # for reproducibility
path_model='./MODEL'
path_log = './log'
nb_epoch=2
batch_size=32
hyperparams_name = 'myCNN'
save_file = os.path.join(path_log, '{}.loss.png'.format(hyperparams_name))

load_pre_model = True 
CAMERA = True
FLIP = True # If you open this swich, pls remember set steps_one_epoch by factor correctly, OR set factor=0
factor = (1/2 +1)
resized_shape = 64 #resized to 64x64

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
	return np.array(sklearn.utils.shuffle(samples))

def gen_data(samples, batch_size):

	def get_pic_and_label(source_path):

		filename = source_path.split('/')[-1]

		current_path='./data/IMG/'+ filename

		img_pil = Image.open(current_path)
		w, h = img_pil.size
		img_cropped = img_pil.crop((0,50,w,h-25))
		img_resized = img_cropped.resize((resized_shape,resized_shape), Image.ANTIALIAS) # resized default to 64x64

		image = np.asarray(img_resized)
		measurement = float(line[3])

		return image, measurement
		
	#offset = 0 
	#print('number of samples is:', len(lines))	
	num_samples = len(samples)
	#train_samples = samples[ :-int(len_samples*0.1) ]
	#valid_samples = samples[ -int(len_samples*0.1): ]

	while True:
				
		for offset in range(0, num_samples, batch_size):			
			images=[]
			measurements=[]
			batch_lines = samples[ offset : offset + batch_size]

			for line in batch_lines:
				# Data Augmentation by right or left camera
				if CAMERA == True:
                                #camera = np.random.choice(['center', 'left', 'right'])
					camera = np.random.choice(['center','left', 'right'])
					if camera == 'left':
                                                camera_path=line[1]
                                                bias=0.22
					elif camera=='right':
						camera_path=line[2]
						bias=-0.22
					elif camera == 'center':
						camera_path=line[0]
						bias=0
				image, measurement = get_pic_and_label(camera_path)
				images.append(image)
				measurements.append(measurement+bias)

				# Data Augmentation by Flip
				if FLIP == True:
					flip_prob = np.random.random()
					if flip_prob > 0:					
						image_flipped = np.fliplr(image)
						measurement_flipped = -measurement
						images.append(image_flipped)
						measurements.append(measurement_flipped)
				
				X=np.array(images)
				y=np.array(measurements)

			yield (X, y)

def build_model():
	model = myCNN(resized_shape=resized_shape)
	model.compile(loss='mse', optimizer='adam')
	model.summary()
	return model

def main():

	print("loading data...")
	samples = load_samples()
	len_samples = samples.shape[0]

	train_samples = samples[ :-int(len_samples*0.1) ]
	valid_samples = samples[ -int(len_samples*0.1): ]
	print('The number of all original (without augmented data) samples is: {} \n,\
		the number of original train_samples is : {} \n \
		the number of original valid_samples is : {} \n'
		.format(len_samples, train_samples.shape[0],valid_samples.shape[0]))

	#train_generator = gen_data(train_samples, batch_size)
	#valid_generator = gen_data(valid_samples, batch_size)
	
	train_generator = gen_data(samples, batch_size)
	valid_generator = gen_data(valid_samples, batch_size)


	fname_param = os.path.join(path_model, '{}.best.h5'.format(hyperparams_name))
	
	early_stopping = EarlyStopping(monitor='val_loss', patience =3, mode='min')
	model_checkpoint = ModelCheckpoint(
	    fname_param, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

	if load_pre_model is True:
		print('*' * 100)
		print('Load pretraining model')
		print('*' * 100)
		model = load_model(os.path.join(path_model, '{}.h5'.format(hyperparams_name)))
	else:
		model = build_model()

	history = model.fit_generator(train_generator, 
		steps_per_epoch = len(train_samples) *(1+factor)// batch_size, 
		validation_data = valid_generator,
		validation_steps = len(valid_samples) *(1+factor)// batch_size,
	    nb_epoch = nb_epoch,
	    verbose = 1,
	    callbacks = [early_stopping, model_checkpoint])

	print('=' * 100)
	
	# list all data in history
	model.save(os.path.join(path_model, '{}.h5'.format(hyperparams_name)), overwrite=True)
	print('{} is saved in {}'.format('{}.h5'.format(hyperparams_name), path_model))
	pickle.dump((history.history), open(os.path.join(path_log, '{}.history.pkl'.format(hyperparams_name)), 'wb'))		
	visual_log(history=history.history, save_path=save_file)
	
	print(history.history.keys())


if __name__ == '__main__':
	main()
	#logfile = os.path.join(path_log, '{}.history.pkl'.format(hyperparams_name))
	#print(logfile)
	#history = pickle.load(open(logfile, "rb"))
	#visual_log(history=history, save_path=path_log)
