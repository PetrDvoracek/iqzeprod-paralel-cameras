import tensorflow as tf
from camera import MultiSourceCV2Gen

from time import time

from utils import timeit

if __name__ == '__main__':
    image_gen = MultiSourceCV2Gen(2, 3)

    model = tf.keras.applications.MobileNetV2(input_shape=(480, 640, 3),
                                                   include_top=False,
                                                   weights=None)

    @timeit()
    def predict(model, batch):
        pred = model.predict(batch)
        print(type(pred))

    timed = predict(model, next(image_gen))
    print(timed.describe())