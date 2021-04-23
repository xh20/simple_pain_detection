# simple_pain_detection

## requirement.txt can be
dlib
opencv-python
numpy
pyrealsense2

### run with real-time pain detection, <make sure that you have a trained svm model!!>  

$ python --camera 1

### run with record your own data  

$ python --create_dataset 1

### train a svm model  

$ python --train 1
