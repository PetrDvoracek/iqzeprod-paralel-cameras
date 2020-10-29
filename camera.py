import cv2
import numpy as np

class MultiCameraCV2Gen:
    def __init__(self, *sources):
        self.sources = sources
        self.captures = [cv2.VideoCapture(x) for x in sources]

    def __iter__(self):
        return self

    def __next__(self):
        return [cap.read()[1] for cap in self.captures]

if __name__ == '__main__':
    gen = MultiCameraCV2Gen(2, 3)
    batch = next(gen)

    image = np.concatenate(batch, axis=1)

    cv2.imshow('joined', image)
    cv2.waitKey(1000)

