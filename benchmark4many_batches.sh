for i in $(seq 17 512) ; do python3 model.py benchmark-trt ./models/mobilenet270x72x3to1-smcli-trt/ --input-shape 270 72 3 --csv models/mobilenet270x72x3to1-smcli-trt-benchmark-batch$i.csv --times 1000 --batch-size $i; done