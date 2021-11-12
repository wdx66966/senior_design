




/home/ubuntu/src/caffe_python_3/build/tools/compute_image_mean -backend=lmdb /home/ubuntu/senior_design/E_Y_E/eye/input/train_lmdb /home/ubuntu/senior_design/E_Y_E/eye/input/mean.binaryproto
/home/ubuntu/src/caffe_python_3/build/tools/caffe train --solver=/home/ubuntu/senior_design/E_Y_E/solver_1.prototxt --weights /home/ubuntu/senior_design/E_Y_E/bvlc_reference_caffenet.caffemodel 2>&1 | tee /home/ubuntu/senior_design/E_Y_E/model_train.log
