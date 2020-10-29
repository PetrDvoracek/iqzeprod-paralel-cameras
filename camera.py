import cv2
import numpy as np
from time import time
import pandas as pd
import seaborn as sns

from utils import timeit
import datetime

class MultiSourceCV2Gen:
    def __init__(self, *sources):
        self.sources = sources
        self.captures = [cv2.VideoCapture(x) for x in sources]
        self.last_capture_time = 0

    def __iter__(self):
        return self

    def __next__(self):
        captures = np.array([cap.read()[1] for cap in self.captures])

        duration_between_catures = time() - self.last_capture_time
        self.last_capture_time = time()

        return captures, duration_between_catures

if __name__ == '__main__':
    img_gen = MultiSourceCV2Gen(2, 3)

    @timeit(times=1000)
    def measure_capture_from_gen(gen):
        next(gen)

    timed = measure_capture_from_gen(img_gen)

    # try:
    #     for img_batch , duration in img_gen:
    #         img2show = np.concatenate(img_batch, axis=1)
    #         cv2.putText(img2show, f'{1 / timed.mean():0.0f} FPS (average)', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, 255)
    #         cv2.putText(img2show, f'{1 / duration:0.0f} FPS (actual)', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, 255)
    #         cv2.imshow('image from specified cameras', img2show)
    #         print(duration)
    #         if cv2.waitKey(1) == ord('q'):
    #             break
    # finally:
    #     cv2.destroyAllWindows()

    print(timed.describe())

    import matplotlib.pyplot as plt
    sns.boxplot(y=timed)
    sns.swarmplot(y=timed, color='.2', size=5, linewidth=2.5)
    plt.show()

