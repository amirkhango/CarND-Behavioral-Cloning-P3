#**Behavioral Cloning** 

##Project Conclustion

Oh my god, I can't believe I can make it and never give up. Nearly I have pay more than 80 hours and finally drive the car successfully. **Firstly, I must conclude the 5 most important aspects which are the biggest tigers on the project for me**

1. Generator technique is a must. Never imagine you can avoid this trick even though you have a great GPU.
This trick really helps you produce augmented training batch on the fly, which greatly saves your GPU and disk storage. Also with it, you can use a bigger batch-size parameter!
2. Balance and diverse dataset is **very, very important** for this project. What does that mean? It means the right turn, left turn, straight route data should at least distribute as normal with the center of steering 0. In this project, to obey Occam's Razor rule, I only use the two augmentation techniques  to make it, i.e., *Flip* and *Recover from Left&Right camera*, without Brightness and Shift skills, to make it work on track 1. Hence I suggest directly use data provided by Udacity, it is distributed well and only need a little augmentation like *Flip* and *Recover from Left&Right camera*.
3. A great and robust model is very important. Here I suggest use the provided model by NVIDA.
4. With balance dataset and a great model (like NVIDIA model), training epoch and lower MSE loss is not very important in this project. Just 3 epochs is enough!
5. The **most tricky phenomenon** for me is that, if your car runs like a drunk, not only your algorithm infulence the driving action, but also maybe the **hard configuration of your PC** does! A case is when I run the code:
```
python drive.py ./MODEL/model.h5
```
My car runs very well and very smoothly from straight line to turn.

But when I run this code, you will see your car start to be a little drunk. 
```
python drive.py ./MODEL/model.h5 run1
```
Why?! I dive into file 'drive.py', I find these lines:
```
steering_angle = float(model.predict(image_array[None, :, :, :], batch_size=1))
....
....
....
....
# save frame
if args.image_folder != '':
    timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
    image_filename = os.path.join(args.image_folder, timestamp)
    image.save('{}.jpg'.format(image_filename))
```

That means after each prediction, your code will save an image frame on the disk. Hence if your PC is not very good, the CPU will consume much resource to sovle the storage problem and has no ability to make the next prediction in time. It will lead the car cannot turn in time, so it behaves a little drunk!

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

[aug]: ./img/augmented_data.png "aug"
[original]: ./img/udacity_original_data_dis.png "original"

[img0]: ./img/original_img.png "raw"
[img1]: ./img/crop_img.png "crop"
[img2]: ./img/crop_resize_img.png "resize"


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create the model
* train.py containing the script to train the model
* drive.py for driving the car in autonomous mode
* .MODEL/model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 is the generated driving video by the script video.py

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```
python drive.py ./MODEL/model.h5
```

####3. Submission code is usable and readable

For clarity, I refer the NVIDIA model and include it in the file model.py and train model in the train.py

The model.py file contains the code of the architecture, by calling nvidia_model(), we can build the prediction network. Actually, this off-the-shelf architecture is referred to NVIDIA's paper. Honestly, at the first time, I have tried to design myself model by calling myCNN() function. After trialing many times, I found myself model is not only larger (about 5MB) than NVIDIA model (about 1MB), but also less robust than NVIDIA model. I.e., NVIDIA model could fit and generalize better with less parameters!

The train.py file will be run for training and saving the NVIDIA model. 

The file 'train.py' shows the pipeline I used for loading, training, validating and saving the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The NVIDA model consists of layers in model.py, which is a very light and clear 'Sequential' model.
Particularly:
1) It accepts data with shape 64x64x3, and then do normaliztion with a lamda layer to rescale data from 0-255 to -0.5-0.5. 
2) Next is the stacked convolutional layers with the different numbers of filters. To avoid overfitting, I use L2 norm to do weight decay with the coefficient 0.001. To realize non-linearity, ReLU layer is utilized after each convolutional layer.
3) Then (line 47 in model.py) I flatten the convolutional feature maps into a vector so that it can be linked with following fully connected layers with different nodes, that is 80->40->16->10. To address overfitting, I use both Dropout (rate=0.5) and L2 norm (0.001) techniques.
4) The last layer is only one node to predicit the steering degree also with L2 norm.

####2. Attempts to reduce overfitting in the model

The model contains dropout layers (0.5) and L2 norm (0.001) in order to reduce overfitting. 

####3. Model parameter tuning

1. The model used an **adam** optimizer, so the learning rate was not tuned manually.
2. **epoch** need not be very large, in my opinion, 3 epochs is enough. For me, at the first training, I set it to 5 to train a trial model and save the weights in the './MODEL/model.h5' file. If the car does not run well, I will load the saved weights and then only train another 2 epochs for saving time.
3. **batch size** is set to 32 considering the GPU size on AWS. Actually, with using Generator, this parameter can be set up to 64.
4. **resized_shape** is set to 64 to fit NVIDIA model.
5. **factor = (2/2 +1)** is a zooming factor defined by myself to indicate how many samples in on epoch for ‘steps_per_epoch’ & ‘validation_steps’
6. **steer_threshold = 0.05 & p_drop_low = 0.8** are also defined by myself. They are combined to determin probability of dropping the sample whose steer is lower than steer_threshold. The aim is to make the data distribute as normal. 
7. I also used Earlystop and Checkpoint method to supervise the training process. But it is unnecessary in this project.

####4. Creation of the Training  & Validation Set

Training data was chosen to keep the vehicle driving on the road. At the starting, I try to collect data as the Udacity suggestion, but it is a tricky process and time consuming. After that, I determined to use data directly provided by Udacity.

After the collection process, I had about 8036 number of data points. I then cropped the unrelated top 50 pixels and bottom 25 pixels for easing the training process. And then resized the image from 160x320x3 to 64x64x3 to fit the model as bellow:

```
img_pil = Image.open(current_path)
		w, h = img_pil.size
		img_cropped = img_pil.crop((0,50,w,h-25))
		img_resized = img_cropped.resize((resized_shape,resized_shape), Image.ANTIALIAS) # resized default to 64x64

```

An original data sample looks like:

![aug image][img0]

After cropped, it looks like:

![aug image][img1]

Then it is reszied:

![aug image][img2]

I finally randomly shuffled the data set and put 10% of the data into a validation set. I.e., 803 for validation and 7233 for training. NB: the real training data are generated on the fly, they are augmented based on 7233 images.

With the same process, I also add below codes in the 'drive.py' to preprocess data:

``` 
 # ========= resized default to 64x64 ====in drive.py=====
        w, h = image.size
        crop_image = image.crop((0,50,w,h-25))
        resize_image = crop_image.resize((resized_shape,resized_shape), Image.ANTIALIAS) 
        image_array = np.asarray(resize_image)
        # ========================================================================
```

The reason why I do not rescale data to -0.5-0.5 here is beacause this preprocess is integrated into keras layer with

```
model.add(Lambda(lambda x: x/255 - 0.5, input_shape = input_shape))
```

####5. Appropriate training data

 Based on the provided data by Udacity, I used a combination of 'Flip' and Recover from Left&Right camera' to augment my data. For details about how I created the training data on the fly:

- I random flip the input image and sterring with probability 50%:
```
if FLIP == True:
					flip_prob = np.random.random()
					if flip_prob > 0.5:					
						image = np.fliplr(image)
						measurement = -measurement
```
- Also I recove from right or left camera to sample data with ’inextreme‘ steering. :
```
# Data Augmentation by right or left camera
				camera = np.random.choice(['center','center','center','center','center','center','left', 'right'])

				if camera == 'left':
					camera_path=sample[1]
					bias=0.22
				elif camera=='right':
					camera_path=sample[2]
					bias=-0.22
				elif camera == 'center':
					camera_path=sample[0]
					bias=0

```
You may ask, why there are much more 'center'? :
```
camera = np.random.choice(['center','center','center','center','center','center','left', 'right'])
```
When I use below code to train my model，I find my car drive a little 'Z-shape' on straight line.
```
camera = np.random.choice([center','left', 'right'])

```
 Hence I increase the sampling of straight images by increaseing the number 'center' to make my model learn much more 'center images'.

 After these augmentation, data distribution looks like as:

 ![aug image][aug]

 The original data provided by Udacity looks like:

 ![original image][original]

 We can see that by augmentation, data become larger and distribute as norm well, both are great for training!


Other augmentation such as Brightness (RGB->HUV->RGB) and Shift by padding are not used, because I wanna obey Occam's Razor rule to train a successful model with less augmentation. However, I learn a lot from the related blog from [Vivek Yadav](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9). With these augmention, model even trained only on track 1 can run well on track 2. But I do not trial this.
