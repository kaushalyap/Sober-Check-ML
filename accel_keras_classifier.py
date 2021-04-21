import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from accel_features import FeatureSet
import numpy as np
from convert_to_tflite import saved_model_to_tflite


def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("label")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def prepare_dataset():
    os.chdir('datasets/accelerometer/accel-labeled/')
    df = pd.read_csv("combined-all-labeled.csv", delimiter=',').drop('group_timestamp', 1)
    print(f"All shape : {df.shape}")
    val_df = df.sample(frac=0.2, random_state=1337)
    train_df = df.drop(val_df.index)
    print(
        "Using %d samples for training and %d for validation"
        % (len(train_df), len(val_df))
    )
    train_ds = dataframe_to_dataset(train_df)
    val_ds = dataframe_to_dataset(val_df)
    for x, y in train_ds.take(1):
        print("Input:", x)
        print("Target:", y)
    train_ds = train_ds.batch(32)
    val_ds = val_ds.batch(32)
    return [train_ds, val_ds]


def train_accel_keras_model():
    train_ds, val_ds = prepare_dataset()
    all_inputs = prepare_inputs()
    all_encoded_features = encode_features(train_ds, all_inputs)
    model = compile_model(train_ds, val_ds, all_encoded_features, all_inputs)
    return model


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def prepare_inputs():
    # Numerical features
    all_inputs = []
    for feature in FeatureSet:
        input = keras.Input(shape=(1,), name=feature.name)
        all_inputs.append(input)
    return all_inputs


def encode_features(train_ds, all_inputs):
    # Numerical features
    encoded_features = []
    print(f'All inputs size : {len(all_inputs)}')
    for feature in FeatureSet:
        encoded = encode_numerical_feature(all_inputs[feature.value], feature.name, train_ds)
        encoded_features.append(encoded)
    all_features = layers.concatenate(encoded_features)
    return all_features


def compile_model(train_ds, val_ds, all_features, all_inputs):
    x = layers.Dense(32, activation="relu")(all_features)
    x = layers.Dropout(0.5)(x)
    output = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(all_inputs, output)
    model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, epochs=80, validation_data=val_ds)
    model.summary()
    # keras.utils.plot_model(model, to_file="accel-keras.png", show_shapes=True, rankdir="LR")
    os.chdir('../../../')
    model.save('models/saved/accel-model/keras')
    return model


def run_tflite_accel_model(tflite_file, accel_input):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(f'Input Details : {input_details}')
    # print(f'Output Details : {output_details[0]["index"]}')
    print(accel_input.shape)
    for feature in FeatureSet:
        input_data = np.array(accel_input[feature.name], dtype=np.float32).reshape(1, 1)
        print(f'Input Details shape : {input_details[feature.value]["shape"]}')
        print(f'Input Data shape : {input_data.shape}, Input Type : {type(input_data)}')
        interpreter.set_tensor(input_details[feature.value]["index"], input_data)

    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])
    print(prediction)
    return 1 if prediction >= 0.5 else 0


def test_model(model):
    test_df = pd.read_csv("test.csv", delimiter=',')
    positive_row = test_df[test_df['label'] == 1].drop(test_df.columns[[0, 1]], axis=1)
    negative_row = test_df[test_df['label'] == 0].drop(test_df.columns[[0, 1]], axis=1)
    positive_item = positive_row.to_dict(orient='records').pop()
    negative_item = negative_row.to_dict(orient='records').pop()
    input_dict = {name: tf.convert_to_tensor([value]) for name, value in negative_item.items()}
    predictions = model.predict(input_dict)
    print(predictions[0][0])


if __name__ == '__main__':
    # model = train_accel_keras_model()
    # test_model(model)
    # saved_model_to_tflite('models/saved/accel-model/keras/', 'accel-k', True)

    converted_model = "models/converted/accel-k.tflite"
    test_df = pd.read_csv('datasets/accelerometer/accel-labeled/test.csv', delimiter=',')
    positive_example = test_df[test_df['label'] == 1].drop(test_df.columns[[0, 1]], axis=1)
    negative_example = test_df[test_df['label'] == 0].drop(test_df.columns[[0, 1]], axis=1)

    prediction = run_tflite_accel_model(converted_model, negative_example)
    if prediction == 1:
        print("Drunk")
    else:
        print("Sober")
