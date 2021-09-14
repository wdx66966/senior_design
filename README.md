'''
Danxu Wang  
Senior Project  
Facial expression reader and player  
+ Files:  
  + 1.Two approches 
     + Facial expression  
     + Eye    
+ Two folder:  
  + FE:  databeases: (FER2013, JAFFE(image), JK+(image))
    + Model Definition "caffenet_train_val_1.prototxt"  
    + Solver Definition "solver_1.prototxt"  
    + Model architecture png   
    + create learning curve script (plot.py) 
  + E_Y_E:  
    + Model Definition "caffenet_train_val_1.prototxt"  
    + Solver Definition "solver_1.prototxt"  
    + Model architecture png  
    + Learning curve png  
    + create learning curve script (plot.py) 


Step:
I should just write a bash.

1. pip install lmdb
2. pip install pydot
3. wget http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
4. python create_lmdb.py
5. /home/ubuntu/src/caffe_python_3/build/tools/compute_image_mean -backend=lmdb /home/ubuntu/senior_design/FE/CK+_JAFFE/input/train_lmdb /home/ubuntu/senior_design/FE/CK+_JAFFE/input/mean.binaryproto
6. python /home/ubuntu/src/caffe_python_3/python/draw_net.py /home/ubuntu/senior_design/FE/train_val_1.prototxt /home/ubuntu/senior_design/FE/model_architecture.png
7. /home/ubuntu/src/caffe_python_3/build/tools/caffe train --solver=/home/ubuntu/senior_design/FE/solver_1.prototxt --weights /home/ubuntu/senior_design/FE/bvlc_reference_caffenet.caffemodel 2>&1 | tee /home/ubuntu/senior_design/FE/model_train.log
