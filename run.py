from camera import MultiSourceCV2Gen

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import cv2
import numpy as np
from scipy import ndimage

import os
import io

import click

def load_model_get_inferentor(model_path, is_trt):
    if is_trt:
        model = tf.saved_model.load(model_path, tags=tag_constants.SERVING)
        infer = build_trt_inferentor(model)
    else:
        model = tf.keras.models.load_model(model_path)
        infer = build_tf_native_inferentor(model)
    return infer

def build_tf_native_inferentor(model):
    return model.predict

def build_trt_inferentor(model):
    signature_keys = list(model.signatures.keys())
    infer = model.signatures['serving_default']
    return infer

def trt_infer_get_cam_pred(infer_func, batch):
    pred = infer_func(tf.constant(batch, dtype=tf.float32))

    list_pred_class_key = [key for key in pred.keys() if 'dense' in key.lower()]
    list_pred_cam_key = [key for key in pred.keys() if 'class_activation' in key.lower()]
    assert len(list_pred_cam_key) == 1
    assert len(list_pred_class_key) == 1
    pred_class_key = list_pred_class_key[0]
    pred_cam_key = list_pred_cam_key[0]

    pred_class = np.array(pred[pred_class_key])
    pred_cam = np.array(pred[pred_cam_key])
    return pred_cam, pred_class

@click.command()
@click.argument('model-path')
@click.option('--n-cams', '-n', default=1, help='Number of cameras to process the images from.')
@click.option('--trt', '-t', is_flag=True, default=False, help='Whether or not model is TensorRT compiled.')
@click.option('--prediction-folder', '-f', default='./predictions')
def main(model_path, n_cams, trt, prediction_folder):
    if not os.path.exists(prediction_folder):
        os.mkdir(prediction_folder)
    print('loading model ...')
    infer = load_model_get_inferentor(model_path, trt)
    print('creating image source ...')
    image_gen = MultiSourceCV2Gen(*list(range(0,n_cams)))
    image_gen.change_res(width=160, height=120)
    image_gen.resize_cam_output(224, 224)

    cv2.namedWindow("stabilized image", cv2.WINDOW_AUTOSIZE );
    print('analyzing images ...')
    try:
        for batch, duration in image_gen:
            img2show = np.concatenate(batch, axis=1)
            img2show = img2show / 255

            cam, pred = trt_infer_get_cam_pred(infer, batch)


            def save_cam(batch, folder):
                class_first_batch = np.transpose(batch, [0,3,1,2])
                for idx, camera in enumerate(class_first_batch):
                    path = os.path.join(folder, str(idx)) + '.txt'
                    if os.path.exists(path):
                        os.remove(path)
                    for pred_class in camera:
                        with open(path, 'a') as f:
                            mem_file = io.StringIO()
                            np.savetxt(mem_file, pred_class, delimiter='\t', fmt='%0.5f')
                            new_data_str = mem_file.getvalue().replace('.', ',')

                            f.write(new_data_str)
                            f.write('\n')
            save_cam(cam, prediction_folder)

            predicted_class = np.argmax(pred, axis=1)
            #predicted_class = 1
            cam2show = np.array([cam[x,:,:,predicted_class[x]] for x in predicted_class])
            cam2show = np.array([ndimage.zoom(x, (224 / x.shape[1], 224 / x.shape[0]), order=0) for x in cam2show])
            cam2show = np.array([cv2.merge([x, x, x]) for x in cam2show])
            cam2show = np.concatenate(cam2show, axis=1)
            print(cam2show.shape, img2show.shape)
            img2show = np.concatenate((img2show, cam2show))



            # cv2.putText(img2show, f'{1 / timed.mean():0.0f} FPS (average)', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, 255)
            cv2.putText(img2show, f'{1 / duration:0.0f} FPS (actual)', (100, 100), cv2.FONT_HERSHEY_COMPLEX, 1, 255)
            # img2show = cv2.resize(img2show, (n_cams * 160, 120))
            cv2.imshow('stabilized image', img2show)
            print(duration)
            if cv2.waitKey(1) == ord('q'):
                raise StopIteration('STOP!')
    finally:
        image_gen.close()
        cv2.destroyAllWindows()





if __name__ == '__main__':
    main()
