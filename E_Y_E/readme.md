source activate caffe_p35
git clone address
cd senior_design
pip install lmdb
pip install pydot
wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
python create_lmdb.py
/home/ubuntu/src/caffe_python_3/build/tools/compute_image_mean -backend=lmdb /home/ubuntu/senior_design/E_Y_E/eye/input/train_lmdb /home/ubuntu/senior_design/E_Y_E/eye/input/mean.binaryproto
/home/ubuntu/src/caffe_python_3/build/tools/caffe train --solver=/home/ubuntu/senior_design/E_Y_E/solver_1.prototxt --weights /home/ubuntu/senior_design/E_Y_E/bvlc_reference_caffenet.caffemodel 2>&1 | tee /home/ubuntu/senior_design/E_Y_E/model_train.log
python /home/ubuntu/senior_design/E_Y_E/plot.py /home/ubuntu/senior_design/FE/model_train.log /home/ubuntu/senior_design/E_Y_E/caffe_model_1_learning_curve.png
