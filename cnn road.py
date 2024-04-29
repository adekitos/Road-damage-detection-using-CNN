import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
#from keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import cv2
import os
import numpy as np


labels = ['D00', 'D20', 'D40'] 
img_size = 224
def get_data(data_dir):
	data = []
	for label in labels:
		path = os.path.join(data_dir, label)
		class_num = labels.index(label)
		for img in os.listdir(path):
			try:
				img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB for mat
				resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
				data.append([resized_arr, class_num])
			except Exception as e:
				print(e)
	return np.array(data)
# Now we can easily fetch our train and validation data.
#train = get_data('../input/traintestsports/Main/train')
#val = get_data('../input/traintestsports/Main/test') 

#C:\Users\Adekitos\Desktop\CNN\Train
train = get_data('C:/Users/Adekitos/Desktop/CNN/Test')
val = get_data('C:/Users/Adekitos/Desktop/CNN/Train')


#Visualize the data
l = []
for i in train:
	if(i[1] == 0):
		l.append("D00")
	elif(i[1] == 1):
		l.append("D20")
	elif(i[1] == 2):
		l.append('D40')
sns.set_style('darkgrid')
sns.countplot(l)
plt.figure(figsize = (5,5))
plt.imshow(train[1][0])
plt.title(labels[train[0][1]])

#data processing and augumentation
x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
	x_train.append(feature)
	y_train.append(label)
	
for feature, label in val:
	x_val.append(feature)
	y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)


#make numpy array
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)
x_val = np.asarray(x_val)
y_val = np.asarray(y_val)



#Data augmentation on the train data:-
datagen = ImageDataGenerator(
featurewise_center=False, # set input mean to 0 over the dataset
samplewise_center=False, # set each sample mean to 0
featurewise_std_normalization=False, # divide inputs by std of the dataset
samplewise_std_normalization=False, # divide each input by its std
zca_whitening=False, # apply ZCA whitening
rotation_range = 30, # randomly rotate images in the range (degrees, 0 to 180)
zoom_range = 0.2, # Randomly zoom image
width_shift_range=0.1, # randomly shift images horizontally (fraction of total width)
height_shift_range=0.1, # randomly shift images vertically (fraction of total height)
horizontal_flip = True, # randomly flip images
vertical_flip=False) # randomly flip images

datagen.fit(x_train)


# Define the model
model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
model.add(MaxPool2D())
model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Conv2D(64, 3, padding="same", activation="relu"))

model.add(MaxPool2D())
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(3, activation="softmax"))
model.summary()

model.compile(optimizer= 'adam' , loss = 'mse', metrics=['accuracy'])
#history = model.fit(x_train, y_train, epochs = 10, validation_data = (x_val,y_val))


train_datagen = ImageDataGenerator(horizontal_flip=True, fill_mode='nearest')
train_generator = train_datagen.flow_from_directory('C:\\Users\\Adekitos\\Desktop\\CNN\\Train', target_size=(224, 224), class_mode='categorical')
#train_generator = train_datagen.flow_from_directory('C:\\Users\\Adekitos\\Desktop\\Olawale binary images\\LOOP\\LOOP_train', target_size=(224, 224), batch_size=100, class_mode='binary')

validation_datagen = ImageDataGenerator(horizontal_flip=True, fill_mode='nearest')
validation_generator = validation_datagen.flow_from_directory('C:\\Users\\Adekitos\\Desktop\\CNN\\Test', target_size=(224, 224), class_mode='categorical')
#validation_generator = validation_datagen.flow_from_directory('C:\\Users\\Adekitos\\Desktop\\Olawale binary images\\LOOP\\LOOP_test', target_size=(224, 224), batch_size=100, class_mode='binary')

#train_generator = np.asarray(train_generator)
#validation_generator = np.asarray(validation_generator)

history = model.fit(train_generator, epochs=100, validation_data=validation_generator)
#history = model.fit_generator(train_generator, steps_per_epoch=10, epochs=10, validation_data=validation_generator)




# Evaluating the result
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(100)
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

          
predictions = model.predict_classes(x_val)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names = ['D00 (class 0)', 'D20 (class 1)', 'D40 (class 1)']))
