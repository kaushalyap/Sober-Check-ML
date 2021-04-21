import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras
from accel_features import FeatureSet
from convert_to_tflite import saved_model_to_tflite
from test_models import run_tflite_accel_model


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    dataframe = dataframe.copy()
    labels = dataframe.pop('label')
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    return ds


def train_accel_tf_model():
    os.chdir('datasets/accelerometer/accel-labeled/')
    df = pd.read_csv("combined-all-labeled.csv", delimiter=',').drop('group_timestamp', 1)
    train, test = train_test_split(df, test_size=0.2)
    train, val = train_test_split(train, test_size=0.2)
    # print(len(train), 'train examples')
    # print(len(val), 'validation examples')
    # print(len(test), 'test examples')
    batch_size = 32
    train_ds = df_to_dataset(train, batch_size=batch_size)
    val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
    test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

    model = compile_train_model(train_ds, val_ds, test_ds, generate_feature_columns())
    os.chdir('../../../')
    model_path = os.getcwd() + "/models/saved/accel-model/tf/"
    model.save(model_path)


def generate_feature_columns():
    feature_columns = []
    # numeric cols
    for feature in FeatureSet:
        feature_columns.append(feature_column.numeric_column(feature.name))
    return feature_columns


def test_feature_set(train_ds):
    for feature_batch, label_batch in train_ds.take(1):
        print('Every feature:', list(feature_batch.keys()))
        print('A batch of x_mean:', feature_batch['x_mean'])
        print('A batch of targets:', label_batch)

    # We will use this batch to demonstrate several types of feature columns
    example_batch = next(iter(train_ds))[0]

    x_mean = feature_column.numeric_column('x_mean')

    # create a feature column
    # and to transform a batch of data
    f_layer = layers.DenseFeatures(x_mean)
    print(f_layer(example_batch).numpy())


def compile_train_model(train_ds, val_ds, test_ds, feature_columns):
    feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
    model = tf.keras.Sequential([
        feature_layer,
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dropout(.1),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile('adam', "binary_crossentropy", metrics=['accuracy'])
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=80)
    model.summary()
    # keras.utils.plot_model(model, to_file="accel-tf.png", show_shapes=True, rankdir="LR")

    loss, accuracy = model.evaluate(test_ds)
    print("Accuracy", accuracy)
    return model


# test_df = pd.read_csv("test.csv", delimiter=',')
# positive_row = test_df[test_df['label'] == 1].drop(test_df.columns[[0, 1]], axis=1)
# negative_row = test_df[test_df['label'] == 0].drop(test_df.columns[[0, 1]], axis=1)
#
#
# positive_item = positive_row.to_dict(orient='records').pop()
# negative_item = negative_row.to_dict(orient='records').pop()
#
#
# input_dict = {name: tf.convert_to_tensor([value]) for name, value in negative_item.items()}
# predictions = model.predict(input_dict)
#
# print(predictions[0][0])

if __name__ == '__main__':
    # train_accel_tf_model()
    # saved_model_to_tflite('models/saved/accel-model/tf/', 'accel', False)

    converted_model = "models/converted/accel.tflite"
    test_df = pd.read_csv('datasets/accelerometer/accel-labeled/test.csv', delimiter=',')
    positive_example = test_df[test_df['label'] == 1].drop(test_df.columns[[0, 1]], axis=1)
    negative_example = test_df[test_df['label'] == 0].drop(test_df.columns[[0, 1]], axis=1)

    prediction = run_tflite_accel_model(converted_model, positive_example)
    # print(prediction)
