# Import libraries:
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
import keras.utils as utils
import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from PIL import Image
import os
import csv

#Prepare X and y train sets:
train_truth = pd.read_csv('/Users/Porto/.spyder-py3/train.truth.csv')
train_truth = train_truth.values

train_X = train_truth[:,0]

train_X_list = []
for x in train_X:
    train_X_list.append(x)
    
train_y = train_truth[:,1]

train_y_list = []
for x in train_y:
    train_y_list.append(x)
    
# Create blank test array:
train_X_input = np.zeros((((48896,64,64,3))))

# Fit the images on the array:
for x in range(0,len(train_X_list)):
    train_X_input[x] = Image.open('/Users/Porto/.spyder-py3/train.rotfaces/train/'+train_X_list[x])
    #counter, since it takes a while:
    print(x)
    
# Scaling data:
for x in range(0,len(train_X_list)):
    train_X_input[x] = train_X_input[x] / 255
    #counter:
    print(x)

# Encoding Categorical Data for the y train set:
    
train_y_input = np.zeros((48896,4))

for x in range(0,len(train_y_list)):
    if train_y_list[x] == 'upright':
        train_y_input[x][0] = 1
    elif train_y_list[x] == 'rotated_right':
        train_y_input[x][1] = 1
    elif train_y_list[x] == 'upside_down':
        train_y_input[x][2] = 1 
    elif train_y_list[x] == 'rotated_left':
        train_y_input[x][3] = 1
    else:
        print("ERROR")
        break
    

        
# CNN model compilation and training:       
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(64, 64, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=800, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dense(units=800, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(units=4, activation='softmax'))

model.compile(optimizer=SGD(lr=0.005), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_X_input, y=train_y_input, epochs=2, batch_size=25)

# Saving the trained CNN:
model.save('image_classifier.h5')

#Prepare the test data:
# Generate blank test array:
test_X_input = np.zeros((((5361,64,64,3))))

# Get photo addresses:
test_dir = os.listdir('/Users/Porto/.spyder-py3/test.rotfaces/test')

# Fit the photos on test array:
for x in range(0,len(test_dir)):
    test_X_input[x] = Image.open('/Users/Porto/.spyder-py3/test.rotfaces/test/'+test_dir[x])
    # Counter:
    print(x)
    
# Scaling test data:
for x in range(0,len(test_dir)):
    test_X_input[x] = test_X_input[x] / 255

# Test the model on the test data:
test_prediction = model.predict(test_X_input)

# Convert test predictions to labels:
test_labels = []
for x in range(0,5361):
    if np.argmax(test_prediction[x]) == 0:
        test_labels.append('upright')
    elif np.argmax(test_prediction[x]) == 1:
        test_labels.append('rotated_right')
    elif np.argmax(test_prediction[x]) == 2:
        test_labels.append('upside_down')
    elif np.argmax(test_prediction[x]) == 3:
        test_labels.append('rotated_left')    

# Convert test arrays to files alongside orientations:
test_csv_array = np.array((test_dir,test_labels))
test_csv_array = np.swapaxes(test_csv_array,0,1)

# Save the test file and label array to csv:
pd.DataFrame(test_csv_array).to_csv("/Users/Porto/.spyder-py3/fixed_rotfaces/test_csv_array.csv", header = None, index = None)

# Translate the prediction into the orientations:
test_orientations = []
for x in range(0,len(test_prediction)):
    test_orientations.append(np.argmax(test_prediction[x]))

# Rotate and save the test set images:
for x in range(0,len(test_dir)):
    img = Image.open('/Users/Porto/.spyder-py3/test.rotfaces/test/'+test_dir[x])
    if test_orientations[x] == 0:
        rotated_img = img
    elif test_orientations[x] == 1:
        rotated_img = img.rotate(90)
    elif test_orientations[x] == 2:
        rotated_img = img.rotate(180)
    elif test_orientations[x] == 3:
        rotated_img = img.rotate(270)
    save_location = '/Users/Porto/.spyder-py3/fixed_rotfaces/'+test_dir[x]+'.png'
    rotated_img.save(save_location)
    
# Fixed images array
    
# Generate blank fixed array:
fixed_array = np.zeros((((5361,64,64,3))))

# Get photo addresses:
fixed_dir = os.listdir('/Users/Porto/.spyder-py3/fixed_rotfaces')

# Fit the photos on fixed array:
for x in range(0,len(fixed_dir)):
    fixed_array[x] = Image.open('/Users/Porto/.spyder-py3/fixed_rotfaces/'+fixed_dir[x])
    # Counter:
    print(x)
    
# Scaling the array:
for x in range(0,len(fixed_dir)):
    fixed_array[x] = fixed_array[x] / 255
    
# Test the model on the fixed array:
fixed_prediction = model.predict(fixed_array)

# Convert predictions to labels:
fixed_labels = []
for x in range(0,5361):
    if np.argmax(fixed_prediction[x]) == 0:
        fixed_labels.append('upright')
    elif np.argmax(fixed_prediction[x]) == 1:
        fixed_labels.append('rotated_right')
    elif np.argmax(fixed_prediction[x]) == 2:
        fixed_labels.append('upside_down')
    elif np.argmax(fixed_prediction[x]) == 3:
        fixed_labels.append('rotated_left')    

# Convert arrays to files alongside orientations:
fixed_csv_array = np.array((fixed_dir,fixed_labels))
fixed_csv_array = np.swapaxes(fixed_csv_array,0,1)

# Save the file and label array to csv:
pd.DataFrame(fixed_csv_array).to_csv("/Users/Porto/.spyder-py3/fixed_rotfaces/fixed_csv_array.csv", header = None, index = None)