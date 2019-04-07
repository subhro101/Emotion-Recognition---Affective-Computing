import os
import glob
import cv2
import numpy
import warnings
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint
from keras.applications import VGG16
from keras.optimizers import RMSprop
from keras.models import Sequential, Model, Input
from keras.layers import Activation, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.utils import multi_gpu_model
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

# Necessary Directories
DATA_DIR = r'/home/CAP4628-2/project2_2/processed_data'
TRAINING_DIR = os.path.join(DATA_DIR, "Training")
VALIDATION_DIR = os.path.join(DATA_DIR, "Validaiton")
TESTING_DIR = os.path.join(DATA_DIR, "Testing")

IMAGE_SIZE = 128
BATCH_SIZE = 32

# Run the model using 4 gpu's since they are available on gavi
# This drastically increases train time
MULTI_GPU = True


# Load images from one of the data sets into memory
# Use this over flow from directory because there is enough room to store everything
# In memory, and this improves speed
def load_images(directory):
    X = []
    Y = []
    for filename in glob.iglob(os.path.join(directory, "Pain/*"), recursive=True):
        X.append(cv2.imread(filename))
        Y.append(1)

    for filename in glob.iglob(os.path.join(directory, "No_pain/*"), recursive=True):
        X.append(cv2.imread(filename))
        Y.append(0)

    # Rescale pixel values to 0.0-1.0, imrpoves training time
    return numpy.divide(numpy.array(X), 255), numpy.array(Y)


# The model we eneded up using to get the most accuracy
# We started with a much larger model and removed layers to reduce overfitting
# This simple model still overfits, but takes much longer to do so.
# Dropout added at each layer to reduce overfitting
def build_custom_model():
    cnn = Sequential()

    # Input layer
    cnn.add(Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.25))

    cnn.add(Conv2D(32, (3, 3), input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Dropout(0.5))

    cnn.add(Flatten())
    cnn.add(Dense(64))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(0.5))

    # Output Layer
    cnn.add(Dense(1))
    cnn.add(Activation('sigmoid'))

    if MULTI_GPU:
        cnn = multi_gpu_model(cnn, gpus=4)
    
    # Use binary crossentropy, since this is a binary model, low learning rate to slow overfitting
    cnn.compile(loss='binary_crossentropy', optimizer=RMSprop(0.00001), metrics=['accuracy'])
    return cnn


# Pre-trained vgg16 model, with a custom set of final layers
# This netted very low accuracy, and high train times.
# We thought that maybe using a pre-trained model could help reduce overfitting,
# by using more general features. But the features turned out to be too general.
def build_vgg16_model():
    input_layer = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

    vgg16 = VGG16(weights="imagenet", include_top=False)
    for layer in vgg16.layers[:5]:
        layer.trainable = False

    vgg16_out = vgg16(input_layer)

    x = Flatten()(vgg16_out)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=input_layer, outputs=x)
    if MULTI_GPU:
        model = multi_gpu_model(model, gpus=4)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# Function for outputing training history charts
# This was very useful in increasing accuracy, because
# We can see when overfitting occurs, and when the model
# is learning/not learning.
def output_history_charts(history):
    pyplot.plot(history.history['acc'])
    pyplot.plot(history.history['val_acc'])
    pyplot.title('Accuracy over Time')
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Test'], loc='upper left')
    pyplot.savefig('acc.png')
    pyplot.clf()

    pyplot.plot(history.history['loss'])
    pyplot.plot(history.history['val_loss'])
    pyplot.title('Loss over Time')
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epoch')
    pyplot.legend(['Train', 'Test'], loc='upper left')
    pyplot.savefig('loss.png')


# Create a network from the custom model
net = build_custom_model()

# Load all image sets into memory
X_train, Y_train = load_images(TRAINING_DIR)
X_val, Y_val = load_images(VALIDATION_DIR)
X_test, Y_test = load_images(TESTING_DIR)

# Print a baseline accuracy for the network
# We can determine if it is acutally improving, or just lucky
print(net.evaluate(X_test, Y_test))

# Checkpoint to save the best model based on performance against the validation set
checkpoint = ModelCheckpoint("best.hdf5", monitor='val_acc', save_best_only=True, mode='max')

# Train the network and save history
hist = net.fit(X_train, Y_train, epochs=25, validation_data=(X_val, Y_val), callbacks=[checkpoint], batch_size=BATCH_SIZE)

# Save the history charts to disk
output_history_charts(hist)

# Load the best model, on validation set
net.load_weights("best.hdf5")

# Print out the metrics
# Evaluate performance based on the testing set
Y_test_pred = net.predict(X_test).round()
print("Confusion Matrix:")
print(confusion_matrix(Y_test, Y_test_pred))
print("Classification Accuracy:", accuracy_score(Y_test, Y_test_pred))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print("Precision:", precision_score(Y_test, Y_test_pred))
    print("Recall:", recall_score(Y_test, Y_test_pred))
print("F1 Score:", f1_score(Y_test, Y_test_pred))
