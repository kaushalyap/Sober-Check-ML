import numpy as np
from skimage import io
from skimage.transform import resize
from convert_to_tflite import saved_model_to_tflite
from selfie_classifier import train_selfie_model
from test_models import run_tflite_model


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

    prediction = run_tflite_model(converted_model, normalized_image)
    if prediction == 1:
        print("Drunk")
    else:
        print("Sober")


def train_convert_model():
    train_selfie_model()
    saved_model_path = "models/saved/selfie-model/"
    saved_model_to_tflite(saved_model_path, False)


if __name__ == '__main__':

    # train_convert_model()
    testing_tflite_model("sober", True)
