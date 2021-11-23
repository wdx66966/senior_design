Step: I should just write a bash.

1. pip install lmdb
2. pip install pydot
3. cd senior_design/FE/
4. wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
5. python create_lmdb.py
6. /home/ubuntu/src/caffe_python_3/build/tools/compute_image_mean -backend=lmdb /home/ubuntu/senior_design/FE/CK+_JAFFE_KDEF/input/train_lmdb /home/ubuntu/senior_design/FE/CK+_JAFFE_KDEF/input/mean.binaryproto
7. python /home/ubuntu/src/caffe_python_3/python/draw_net.py /home/ubuntu/senior_design/FE/train_val_1.prototxt /home/ubuntu/senior_design/FE/model_architecture.png
8. /home/ubuntu/src/caffe_python_3/build/tools/caffe train --solver=/home/ubuntu/senior_design/FE/solver_1.prototxt --weights /home/ubuntu/senior_design/FE/bvlc_reference_caffenet.caffemodel 2>&1 | tee /home/ubuntu/senior_design/FE/model_train.log
9. python /home/ubuntu/senior_design/FE/plot.py /home/ubuntu/senior_design/FE/model_train.log /home/ubuntu/senior_design/FE/caffe_model_1_learning_curve.png
