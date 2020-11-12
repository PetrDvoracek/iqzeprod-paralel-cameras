import click
from time import time

import numpy as np

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.compiler.tensorrt import trt_convert as trt

from utils import timeit

@click.command('save')
@click.argument('input-shape', nargs=3, type=(int, int, int))
@click.argument('classes', type=int)
@click.argument('save-path')
def save(input_shape, classes, save_path, channels):
    model = MobileNetV2(input_shape=input_shape, include_top=False, classes=classes)
    model.save(save_path)

@click.command('convert')
@click.argument('model-path', type=click.Path(exists=True))
@click.argument('output-path')
@click.option('--precision', default='fp32', type=click.Choice(['fp32', 'fp16', 'int8']))
@click.option('--engine', default=False, is_flag=True)
def convert(model_path, output_path, precision, engine):
    if precision == 'fp16':
        precision_mode = trt.TrtPrecisionMode.FP16
    elif precision == 'int8':
        precision_mode = trt.TrtPrecisionMode.int8
    else: 
        precision_mode = trt.TrtPrecisionMode.FP32
    
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode=precision_mode,
                                                            max_workspace_size_bytes=5000000000)

    converter = tf.experimental.tensorrt.Converter(
        input_saved_model_dir=model_path, conversion_params=conversion_params
    )

    converter.convert()
    converter.save(output_saved_model_dir=output_path)


@click.command('benchmark-trt')
@click.argument('model-path')
@click.option('--batch-size', default=1)
@click.option('--times', default=100)
@click.option('--skip', default=10)
@click.option('--csv', default='')
@click.option('--input-shape', nargs=3, type=(int, int, int))
def benchmark_trt(model_path, batch_size, times, skip, csv, input_shape):

    model = tf.saved_model.load(model_path, tags=tag_constants.SERVING)
    signature_keys = list(model.signatures.keys())
    infer = model.signatures['serving_default']

    images = np.random.rand(batch_size, *input_shape)
    images = tf.constant(images, dtype=tf.float32)
    print(f'batch size: {batch_size}')

    @timeit(times=times, skip=skip)
    def measure():
        infer(images)

    timed = measure()
    print(timed.describe())
    if csv:
        timed.to_csv(csv)

    # model = tf.saved_model.load(model_path, tags=tag_constants.SERVING)
    # signature_keys = list(model.signatures.keys())
    # infer = model.signatures['serving_default']

    # images = np.array([np.zeros((224, 224, 3)) for _ in range(0, batch_size)])
    # images = tf.constant(images, dtype=tf.float32)
    # print(f'batch size: {batch_size}')

    # for _ in range(0, times):
    #     before = time()
    #     infer(images)
    #     print(f'inference time: {time() - before}')


@click.group()
def cli():
    pass

if __name__ == '__main__':
    cli.add_command(save)
    cli.add_command(benchmark_trt)
    cli.add_command(convert)
    cli()
