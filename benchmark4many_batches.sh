for i in $(seq 1 6) ; do python3 model.py benchmark-trt models/unet_mobilenet-trt-more-space --input-shape 128 160 3 --csv models/unet_mobilenet-trt-more-space-random-images-batch$i.csv --times 1000 --batch-size $i; done
