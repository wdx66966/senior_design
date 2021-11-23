Step: I should just write a bash.

1. pip install --upgrade pip
2. pip install lmdb
3. pip install pydot
4. cd senior_design/FE/
5. wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
6. python create_lmdb.py
7. /home/ubuntu/src/caffe_python_3/build/tools/compute_image_mean -backend=lmdb /home/ubuntu/senior_design/FE/CK+_JAFFE_KDEF/input/train_lmdb /home/ubuntu/senior_design/FE/CK+_JAFFE_KDEF/input/mean.binaryproto
8. python /home/ubuntu/src/caffe_python_3/python/draw_net.py /home/ubuntu/senior_design/FE/train_val_1.prototxt /home/ubuntu/senior_design/FE/model_architecture.png
9. /home/ubuntu/src/caffe_python_3/build/tools/caffe train --solver=/home/ubuntu/senior_design/FE/solver_1.prototxt --weights /home/ubuntu/senior_design/FE/bvlc_reference_caffenet.caffemodel 2>&1 | tee /home/ubuntu/senior_design/FE/model_train.log
10. python /home/ubuntu/senior_design/FE/plot.py /home/ubuntu/senior_design/FE/model_train.log /home/ubuntu/senior_design/FE/caffe_model_1_learning_curve.png
