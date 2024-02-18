# Originally taken from https://github.com/jasonchang0/SoBr/blob/master/bin/convNetKerasLarge.py
# and modified as necessary

import os
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers
from skimage import io
from skimage.transform import resize
import numpy as np
import tensorflow as tf
import random
import glob
from tensorflow import keras

no_samples = 4000
batch_size = 32
epochs = 80

n_image_rows = 106
n_image_cols = 106
n_channels = 3


def train_selfie_model():
    # random_seed = 1
    # tf.random.set_seed(random_seed)
    # np.random.seed(random_seed)

    x_train, y_train = prepare_train_set()

    print(len(x_train))
    print(len(y_train))

    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.30, random_state=42)

    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([1, 1, 1])
    x_train = x_train.astype('float')
    x_test = x_test.astype('float')
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]

    model = compile_model()

    model.summary()
    # keras.utils.plot_model(model, to_file="selfie.png", show_shapes=True, rankdir="LR")

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)

    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])

    model_path = os.getcwd() + "/models/saved/selfie-model/"
    model.save(model_path)


def prepare_train_set():
    positive_samples = glob.glob('datasets/frontal-faces/train/drunk/*')[0:no_samples]
    negative_samples = glob.glob('datasets/frontal-faces/train/sober/*')[0:no_samples]
    negative_samples = random.sample(negative_samples, len(positive_samples))
    x_train = []
    y_train = []
    for i in range(len(positive_samples)):
        x_train.append(resize(io.imread(positive_samples[i]), (n_image_rows, n_image_cols)))
        y_train.append(1)
        if i % 1000 == 0:
            print('Reading positive image number ', i)
    for i in range(len(negative_samples)):
        x_train.append(resize(io.imread(negative_samples[i]), (n_image_rows, n_image_cols)))
        y_train.append(0)
        if i % 1000 == 0:
            print('Reading negative image number ', i)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train, y_train


def compile_model():
    model_input_shape = (n_image_rows, n_image_cols, n_channels)
    model = Sequential()
    model.add(
        Conv2D(8, kernel_size=(3, 3), activation='relu', strides=(1, 1), padding='same', input_shape=model_input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Dropout(0.25))
    model.add(Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Conv2D(8, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    # single output neuron
    model.add(Dense(1, activation='sigmoid'))
    sgd = optimizers.SGD(lr=.001, momentum=0.9, decay=0.000005, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model


def run_tflite_selfie_model(tflite_file, test_image):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(f'Input Details shape : {input_details[0]["shape"]}')
    print(f'Input Data shape : {test_image.shape}, Data Type : {type(test_image)}')

    interpreter.set_tensor(input_details[0]["index"], test_image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])
    # print(prediction)
    return 1 if prediction >= 0.5 else 0


def testing_tflite_model(sample_type, quantized):
    converted_model = "models/converted/selfie.tflite"
    if quantized:
        converted_model = "models/converted/selfie-quantized.tflite"

    drunk_image_path = "datasets/frontal-faces/remaining/drunk/haar_3_img933LR_noop_addLNp_addGNp_LContrast.png"
    sober_image_path = "datasets/frontal-faces/remaining/sober/haar_2_img623UR_ABlur_addLNp_MBlur.png"
    img = io.imread(drunk_image_path)
    if sample_type == "sober":
        img = io.imread(sober_image_path)

    resized = resize(img, (106, 106)).astype('float32')
    test_image = np.expand_dims(resized, axis=0)
    normalized_image = test_image - 0.5

    print(type(normalized_image))
    prediction = run_tflite_selfie_model(converted_model, normalized_image)
    if prediction == 1:
        print("Drunk")
    else:
        print("Sober")


def train_convert_model():
    train_selfie_model()
    # saved_model_path = "models/saved/selfie-model/"
    # saved_model_to_tflite(saved_model_path, 'selfie', False)


if __name__ == '__main__':

    # train_convert_model()
    testing_tflite_model("drunk", True)
