import tensorflow as tf


def saved_model_to_tflite(model_path, quantize):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    model_saving_path = "models/converted/selfie.tflite"
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        model_saving_path = "models/converted/selfie-quantized.tflite"
    tflite_model = converter.convert()
    with open(model_saving_path, 'wb') as f:
        f.write(tflite_model)
