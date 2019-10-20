#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 15:20:13 2017

@author: zhou
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:23:03 2017

@author: zhou
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 22:14:27 2017

@author: zhou
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,SeparableConv2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.layers import Activation, Dropout, Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
#from keras.utils import np_utils
#from  keras.callbacks import ModelCheckpoint, Callback  
from keras import backend as K
import matplotlib.pyplot as plt
#import functools
import matplotlib.lines as mlines
import seaborn as sns
import keras
# dimensions of our images.

#top5_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)

#top5_acc.__name__ = 'top5_acc'
#img_width, img_height = 64, 64

###############################class43###################################
train_data_dir = 'data/gtsrb_43/train'
validation_data_dir = 'data/gtsrb_43/test'

###############################vcifar100###################################
#train_data_dir = 'data/vcifar-100-jpg/train_imgs/18'
#validation_data_dir = 'data/vcifar-100-jpg/test_imgs/18'

###############################end#######################################

########################################class43############################
nb_train_samples = 1500*43
nb_validation_samples = 60*43
######################################end#################################

########################################class5############################
#nb_train_samples = 500*5
#nb_validation_samples = 100*5
######################################end#################################

epochs = 10
batch_size = 16

img_width, img_height = 64, 64

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def leakyCReLU(x):
	x_pos = K.relu(x, .0)
	x_neg = K.relu(-x, .0)
	return K.concatenate([x_pos, x_neg], axis=1)
	
def leakyCReLUShape(x_shape):
	shape = list(x_shape)
	shape[1] *= 2
	return tuple(shape)

def conv_block(x_input, num_filters,pool=True,activation='relu',init='orthogonal'):
	
	x1 = Conv2D(num_filters,3,3, border_mode='same',W_regularizer=l2(1e-4),init=init)(x_input)
	x1 = BatchNormalization(axis=1,momentum=0.995)(x1)
	if activation == 'crelu':
		x1 = Lambda(leakyCReLU, output_shape=leakyCReLUShape)(x1)
	else:
		x1 = LeakyReLU(.01)(x1)
	# x1 = Convolution3D(num_filters,3,3,3, border_mode='same',W_regularizer=l2(1e-4))(x1)
	# x1 = BatchNormalization(axis=1)(x1)
	# x1 = LeakyReLU(.1)(x1)
	
	if pool:
		x1 = MaxPooling2D()(x1)
	x_out = x1
	return x_out

model = Sequential()


#model = Sequential()


####################################################### VGG16 ####################################################
model.add(Conv2D(64,(3,3),strides=(1,1),input_shape=(64,64,3),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(64,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(128,(3,2),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(128,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(Conv2D(512,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
model.add(MaxPooling2D(pool_size=(2,2)))  
model.add(Flatten())  
#model.add(Dense(4096,activation='relu')) 
model.add(Dense(64,activation='relu'))   
model.add(Dropout(0.5))  
model.add(Dense(256,activation='relu'))  
model.add(Dropout(0.5))  
model.add(Dense(10,activation='softmax'))  
#adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
model.summary()  
################################################################ end ##################################################


#model.add(Conv2D(100, (7, 7), input_shape=input_shape))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(150, (4, 4)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#model.add(Conv2D(250, (4, 4)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(128, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Flatten())
#model.add(Dense(300))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
#
#model.add(Dense(8))
#model.add(Activation('softmax'))
#model.add(Dense(1))
#model.add(Activation('sigmoid'))

################################################################### ALEX #######################################################

#model.add(Conv2D(96,(11,11),strides=(4,4),input_shape=(64,64,3),padding='valid',activation='relu',kernel_initializer='uniform'))  
#model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))  
#model.add(Conv2D(256,(5,5),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
#model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(384,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(Conv2D(256,(3,3),strides=(1,1),padding='same',activation='relu',kernel_initializer='uniform'))  
#model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))  
#model.add(Flatten())  
##model.add(Dense(4096,activation='relu'))  
#model.add(Dense(64,activation='relu'))  
#model.add(Dropout(0.5))  
#model.add(Dense(64,activation='relu'))  
#model.add(Dropout(0.5))  
#model.add(Dense(8,activation='softmax'))  
##adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
##rmsprop = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-6)
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])  
#model.summary() 

################################################################## end ########################################################

################################################################### FC #######################################################
#model.add(Conv2D(32,kernel_size=(3,3),strides=(2,2), padding='same', input_shape=input_shape))
##model.add(Conv2D(100,kernel_size=(3,3),strides=(2,2), padding='same', input_shape=input_shape))
##model.add(Lambda(leakyCReLU, output_shape=leakyCReLUShape))
##model.add(Conv2D(32,kernel_size=(3,3),strides=(2,2), padding='same', input_shape=input_shape))
##x1 = conv_block(16, activation='crelu') 
##model.add(Conv2D(32, (3, 3), input_shape=input_shape))
##model.add(Activation('relu'))
##model.add(MaxPooling2D(pool_size=(2, 2)))
##*****************************************************************************************************
##model.add(DepthwiseConv2D(32,strides=(2,2),padding='same'))
#model.add(SeparableConv2D(32,kernel_size=(3,3), strides=(2,2),padding='same'))
##model.add(SeparableConv2D(100,kernel_size=(3,3), strides=(2,2),padding='same'))
##model.add(MaxPooling2D(pool_size=(1, 1)))
#model.add(BatchNormalization())
##model.add(Lambda(leakyCReLU, output_shape=leakyCReLUShape))
#model.add(Activation('relu'))
#
##model.add(Conv2D(32,kernel_size=(1,1),strides=(1,1), padding='same'))
###model.add(MaxPooling2D(pool_size=(1, 1)))
##model.add(BatchNormalization())
##model.add(Activation('relu'))
##model.add(Conv2D(32, (1, 1)))
##model.add(BatchNormalization())
##model.add(Conv2D(32, (3, 3), input_shape=input_shape))
##model.add(Activation('relu'))
##model.add(MaxPooling2D(pool_size=(2, 2)))
##outputs 13 ch
##*****************************************************************************************************
##model.add(SeparableConv2D(32,kernel_size=(3,3), strides=(2,2),padding='same'))
#model.add(Conv2D(32, (3, 3)))
##model.add(Conv2D(100, (3, 3)))
#model.add(Lambda(leakyCReLU, output_shape=leakyCReLUShape))
##model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#
#
#model.add(Conv2D(64, (3, 3)))
##model.add(Conv2D(200, (3, 3)))
#model.add(Lambda(leakyCReLU, output_shape=leakyCReLUShape))
##model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

################################################################## end ########################################################


# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)
#test_datagen = np_utils.to_categorical(test_datagen, 43)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

#validation_generator = np_utils.to_categorical(validation_generator, 43)
#filepath="weights-improvement-{epoch:02d}-{accuracy:.2f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True,mode='max')
#callbacks_list = [checkpoint]



result = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

with open('baseline.txt','w') as f:
    f.write(str(result.history))

model.save('baseline.h5')

#plt.figure
#plt.plot(result.epoch,result.history['acc'],label="acc")
#plt.plot(result.epoch,result.history['val_acc'],label="val_acc")
#plt.scatter(result.epoch,result.history['acc'],marker='*')
#plt.scatter(result.epoch,result.history['val_acc'])
#plt.legend(loc='under right')
#plt.show()
#plt.figure
#plt.plot(result.epoch,result.history['loss'],label="loss")
#plt.plot(result.epoch,result.history['val_loss'],label="val_loss")
#plt.scatter(result.epoch,result.history['loss'],marker='*')
#plt.scatter(result.epoch,result.history['val_loss'],marker='*')
#plt.legend(loc='upper right')
#plt.show()

sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure
plt.plot(result.epoch,result.history['acc'],label="Training",c='r')
plt.plot(result.epoch,result.history['val_acc'],label="Validation",linestyle="--",c='b')
plt.scatter(result.epoch,result.history['acc'],s=150,c='r')
plt.scatter(result.epoch,result.history['val_acc'],marker='*',s=200,c='b')
plt.xlabel('Number of Epochs')
plt.ylabel('Accuracy')
ta = mlines.Line2D([], [], color='red', marker='.',
                          markersize=15, label='Training')
va = mlines.Line2D([], [], color='blue', marker='*',linestyle="--",
                          markersize=15, label='Validation')

plt.legend(handles=[ta,va],loc='lower right',fontsize=12,frameon=True,edgecolor='black')
#plt.grid()
#plt.savefig("gtsrb_fc_8_1_d64_sgd_doublep_acc.eps", format='eps', dpi=1000)
plt.savefig("baseline0.eps", format='eps', dpi=1000)

sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.clf()
plt.figure
plt.plot(result.epoch,result.history['loss'],label="Training",c='r')
plt.plot(result.epoch,result.history['val_loss'],label="Validation",linestyle="--",c='b')
plt.scatter(result.epoch,result.history['loss'],s=150,c='r')
plt.scatter(result.epoch,result.history['val_loss'],marker='*',s=200,c='b')

#plt.plot(result.epoch,result.history['loss'],label="Training",linestyle="--",c='b')
#plt.plot(result.epoch,result.history['val_loss'],label="Validation",c='r')
#plt.scatter(result.epoch,result.history['loss'],marker='*', s=200,c='b')
#plt.scatter(result.epoch,result.history['val_loss'], s=150,c='r')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
tl = mlines.Line2D([], [], color='red', marker='.',
                          markersize=15, label='Training')
vl = mlines.Line2D([], [], color='blue', marker='*',linestyle="--",
                          markersize=15, label='Validation')
#tl = mlines.Line2D([], [], color='blue', marker='*',linestyle="--",
#                          markersize=15, label='Training')
#vl = mlines.Line2D([], [], color='red', marker='.',
#                          markersize=15, label='Validation')
plt.legend(handles=[tl,vl],loc='upper right',fontsize=12,frameon=True,edgecolor='black')
#plt.grid()
#plt.savefig("gtsrb_fc_8_1_d64_sgd_doublep_loss.eps", format='eps', dpi=1000)
plt.savefig("baseline.eps", format='eps', dpi=1000)
plt. close(0)
## summarize history for loss
#plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
#plt.title('model loss')
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.legend(['train', 'test'], loc='upper left')
#plt.show()


#model.save('fc1_gtsrb_vgg16_d256_sgd.h5')
#model.save_weights('GB_first_try.h5')
