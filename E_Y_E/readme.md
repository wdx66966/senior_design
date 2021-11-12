1. source activate caffe_p35
2. git clone address
3. cd senior_design
4. pip install lmdb
5. pip install pydot
6. wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
7. python create_lmdb.py
8./home/ubuntu/src/caffe_python_3/build/tools/compute_image_mean -backend=lmdb /home/ubuntu/senior_design/E_Y_E/eye/input/train_lmdb /home/ubuntu/senior_design/E_Y_E/eye/input/mean.binaryproto
9./home/ubuntu/src/caffe_python_3/build/tools/caffe train --solver=/home/ubuntu/senior_design/E_Y_E/solver_1.prototxt --weights /home/ubuntu/senior_design/E_Y_E/bvlc_reference_caffenet.caffemodel 2>&1 | tee /home/ubuntu/senior_design/E_Y_E/model_train.log
10.python /home/ubuntu/senior_design/E_Y_E/plot.py /home/ubuntu/senior_design/FE/model_train.log /home/ubuntu/senior_design/E_Y_E/caffe_model_1_learning_curve.png
