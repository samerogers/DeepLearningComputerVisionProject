# DeepLearningComputerVisionProject

Requirements: Python 2.7, TensorFlow, Pillow, Numpy, OpenCV

This project attempts to integrate YOLOv3's object detection framework with an LSTM layer for tracking.

YOLOv3 outputs are generated OFFLINE by running genYOLOdetections.py. This will invoke a TensorFlow implementation of YOLOv3 to generate detections for videos in the 'data' folder. Videos should be organized such as individual frames in a folder called 'img'. Additional the main video folder should contain 'groundtruth_rect.txt'. Ground truth labels and YOLO detections will have the form [X0,Y0, Width, Height]. YOLO weights can be found at https://pjreddie.com/darknet/yolo/

To train the LSTM layer simply run training.py, and for testing run testing.py. These files will work on data from the 'MOT_syntheticdata' folder. Training and testing sets are partitioned into their respective folders. Numpy arrays for YOLO bounding boxes and ground truth labels should be placed in the bboxes folder for each video. The numpy array for feature maps should be placed in the features folder for each video.
