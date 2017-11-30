import matplotlib.pyplot as plt

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

def plot_distribution(data_label):
	plt.figure()
	plt.hist(data_label, bins='15')
	plt.title('data distribution')
	plt.ylabel('numbers_')
	plt.xlabel('steer')
