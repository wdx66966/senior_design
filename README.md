'''
Danxu Wang  
Senior Project  
Facial expression reader and player 

The project is using CNN to train a workable neural network model from three databases (FER2013, JAFFEJK+ and KDEF) and over 30000+ image. It is performed and implemented in Raspberry Pi 4B, real time camera and 5 ” touchscreen using Python, OpenCV and several necessary packages. 
The program takes a snapshot from the camera and determines whether the eyes are open or close while detecting the face expression in real time. 
Project implements a workable script with GUI in RPI that performs the following  functions:
+ Face Detection using pre-trained CNN model
+ Recognition of seven facial expression recognition.
+ Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise.
+ Eye state detection (open & closed).
+ Audio output to alert of the expression/ or state of eyes( open/closed) using the speech synthesis 
+ Intel NCS 2 is for boosting the processing time
+ All detections can be done with multiple persons in the frame

+ Files:  
  + 1.Two folder 
     + Facial expression  
     + Eye    
+ abstract:  
  + FE:  databeases: (FER2013, JAFFE(image), JK+(image),KDEF(front face))
    + Model Definition "caffenet_train_val_1.prototxt"  
    + Solver Definition "solver_1.prototxt"   
    + create learning curve script (plot.py) 
  + E_Y_E:  
    + Model Definition "caffenet_train_val_1.prototxt"  
    + Solver Definition "solver_1.prototxt"  
    + Model_architecture png
    + create learning curve script (plot.py) 
