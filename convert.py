from tensorflow.python.compiler.tensorrt import trt_convert as trt
import tensorflow as tf

import click

def convert(model_path, output_path, tf1, precision, max_workspace_size, min_segment_size, saved_model_tags, build, batch_shape):
    if not tf1:
        params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
            max_workspace_size_bytes=max_workspace_size,
            precision_mode=precision,
            minimum_segment_size=min_segment_size)
        converter = trt.TrtGraphConverterV2(
            input_saved_model_dir=model_path,
            input_saved_model_tags=saved_model_tags,
            conversion_params=params)
        try:
            converter.convert()
        except Exception as e:
            raise RuntimeError(
                '{}. Just try passing "--tf1".'.format(e))
        if build or batch_shape[0]:
            def reference_data_gen():
                inp1 = tf.random.normal(size=batch_shape).astype(tf.float32)
                inp2 = tf.random.normal(size=batch_shape).astype(tf.float32)
                yield (inp1, inp2)
            converter.build(reference_data_gen)
        converter.save(output_saved_model_dir=output_path)
    else:
        trt.create_inference_graph(
            None,
            None,
            max_batch_size=1,
            max_workspace_size_bytes=max_workspace_size,
            precision_mode=precision,
            minimum_segment_size=minimum_segment_size,
            is_dynamic_op=True,
            input_saved_model_dir=model_path,
            input_saved_model_tags=saved_model_tags,
            output_saved_model_dir=output_path)

@click.command()
@click.argument('model-path')
@click.argument('output-path')
@click.option('--tf1', is_flag=True, default=False, help='Use TensorFlow 1.x model.')
@click.option('--precision', type=click.Choice(['fp32', 'fp16', 'int8']), default='fp32')
@click.option('--max-workspace-size', default= (2 << 20), help='The maximum GPU temporary memory which the TRT engine can use at execution time in bytes.')
@click.option('--min-segment-size', default=3, help='The minimum number of nodes required for a subgraph to be replacedin a TensorRT node.')
@click.option('--saved-model-tags', default='serve', help='If multiple tags, separate them by \',\'')
@click.option('--build', default=False, help='Build the TensorRT engine (speed up the first inference)')
@click.option('--batch-shape', default=(0,0,0,0), nargs=4, type=(int,)*4, help='Use instead --build. Specify the input shape, typicaly batch_n, img_width, img_height, img_channels (--batch-shape 2 16 16 3 to get batch size of 2 on image 16x16x3')
def main(model_path, output_path, tf1, precision, max_workspace_size, min_segment_size, saved_model_tags, build, batch_shape):
    saved_model_tags = saved_model_tags.split(',')
    convert(model_path, output_path, tf1, precision, max_workspace_size, min_segment_size, saved_model_tags, build, batch_shape)


if __name__ == '__main__':
    main()
  