import tensorflow as tf


def saved_model_to_tflite(model_path, model_name, quantize):
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    model_saving_path = "models/converted/"+model_name+".tflite"
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        model_saving_path = "models/converted/"+model_name+"-quantized.tflite"
    tflite_model = converter.convert()
    with open(model_saving_path, 'wb') as f:
        f.write(tflite_model)
