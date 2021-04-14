import tensorflow as tf


def run_tflite_model(tflite_file, test_image):
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]["index"], test_image)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]["index"])
    return 1 if predictions >= 0.5 else 0  # 1 = sober, 0 = drunk
