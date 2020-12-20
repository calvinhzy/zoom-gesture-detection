# zoom-gesture-detection
We propose to engage the people to interact on Zoom calls in a completely new way, the good old traditional way, 
which is to turn on the videos and make gestures which not only would not disturb the current flow of the presentation (no voice needed), but also engages the audience in a much better way than simply clicking a button. 

# Overview
We were using the coordinates of the 21 key points of each hand outputed from OpenPose directly after normalization. 
They each have a confidence score attached where we have done some thresholding. 
If less than 40% of the key points of the hand has more than 30% confidence (10% higher than the default for hand keypoint rendering), we do not include that sample, this is to combat the case where we have the camera running all the time in zoom and we do not want to spam interactions when the user is not actually trying to make a gesture. 
We can make a further threshold on the output so if only the user holds the gesture for a few seconds, then we trigger the interaction. 

In addition to the key points, we have connected the edges between these key points, creating edge vectors. We also included distance between fingertips to the palm, the distance between each fingers, and index of highest and lower points. These features were helped in the final classification.


# Installation and run
We use the openpose demo code and Python API. https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases
Make sure you have the requirements for running OpenPose, most notably CUDA and OpenCV.
Just replace the openpose/python folder in the downloaded code and you can start making gesture classification.

You would need a webcam. Run the 07_hand_from_image.py. Feel free to checkout OpenPose's documentation for more optimization.

# Training your own data
You can use the classify.py and deeplearning.py for training your data. To collect the data, you can change the code in 07_hand_from_image.py.
