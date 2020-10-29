from time import time
import pandas as pd

def timeit(times=10, skip=2):
    assert times > skip
    def decorator(func):
        def wrapped(*args, **kwargs):
            durations = []
            for i in range(0, times):
                before = time()
                func(*args, **kwargs)
                if i > skip:
                    durations.append(time() - before)
            return pd.Series(durations)
        return wrapped
    return decorator

if __name__ == '__main__':
    @timeit()
    def test_fn():
        print('test')

    test_fn()
