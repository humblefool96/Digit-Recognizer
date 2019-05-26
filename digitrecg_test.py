import numpy as np
import pandas as pd
import seaborn as sns
import warnings

np.random.seed(2)
warnings.filterwarnings('ignore')
sns.set(style='white', context='notebook', palette='deep')

#load data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#matrix of features and predicted value
X = train.drop(labels = ['label'], axis = 1)
Y = train['label']

#visualizing count plot
sns.countplot(Y)
print(Y.value_counts())

#checking for missing values
print(X.isnull().sum())
print(X.isnull().any().describe())
print(test.isnull().any().describe())

#normalize the data
X = X / 255.0
test = test / 255.0

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X = X.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
from keras.utils.np_utils import to_categorical
Y = to_categorical(Y, num_classes = 10)

# Split the train and the validation set for the fitting
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test  = train_test_split(X, Y, test_size = 0.1, random_state=2)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout  
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

#intializing CNN
# my CNN architechture is In -> [[Conv2D->relu]*2 -> MaxPool2D -> Dropout]*2 -> Flatten -> Dense -> Dropout -> Out

digit_classifier = Sequential()

digit_classifier.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
digit_classifier.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
digit_classifier.add(MaxPool2D(pool_size=(2,2)))
digit_classifier.add(Dropout(0.25))


digit_classifier.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
digit_classifier.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
digit_classifier.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
digit_classifier.add(Dropout(0.25))

digit_classifier.add(Flatten())
digit_classifier.add(Dense(256, activation = "relu"))
digit_classifier.add(Dropout(0.5))
digit_classifier.add(Dense(10, activation = "softmax"))

# Define the optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
digit_classifier.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

epochs = 5 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

# Fit the model
history = digit_classifier.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 2, steps_per_epoch=x_train.shape[0] // batch_size
                              , callbacks=[learning_rate_reduction])


# Predict the values from the validation dataset
Y_pred = digit_classifier.predict(x_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred,axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test,axis = 1) 

# predict results
results = digit_classifier.predict(test)

# select the indix with the maximum probability
results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)








