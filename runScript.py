import cv2
import tensorflow as tf
import argparse
import imutils

CATEGORIES = ["dosa","idli","meduvada","nosouth"]
def prepare(file):
    IMG_SIZE = 50
    new_array = cv2.resize(file, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("CNN.model")

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
image = cv2.imread(args["image"],cv2.IMREAD_GRAYSCALE)
image = prepare(image)

prediction = model.predict([image])
prediction = list(prediction[0])
print(prediction)
print(CATEGORIES[prediction.index(max(prediction))])
