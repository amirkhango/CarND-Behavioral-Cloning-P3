import csv
import cv2
import numpy as np

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
##X_train=np.array(images)[:1002]
##y_train=np.array(measurements)[:1002]

 X_train=np.array(images)
 y_train=np.array(measurements)
 
 return X_train, y_train

def build_model():
 model = myCNN()
 model.complie(loss='mse', optimizer='adam')
 model.summary()
 return model

def  main():
 print("loading data...")
 X_train, y_train = load_data()

 
 model = myCNN()
 early_stopping = EarlyStopping(monitor='val_mse', patience =5, mode='min')
 model_checkpoint = ModelCheckpoint(
	    fname_param, monitor='val_mse', verbose=0, save_best_only=True, mode='min')
 
 model.fit(X_train, y_train, validation_split=0.2,
        shuffle=True, nb_epoch=2,
        callbacks=[early_stopping, model_checkpoint],)

 model.save('model.h5')

if __name__ == '__main__':
    main()
