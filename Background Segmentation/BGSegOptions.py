import numpy as np
import cv2 as cv

cap = cv.VideoCapture('./test.mp4')
bg_sub_method = {
    "mog":  [cv.bgsegm.createBackgroundSubtractorMOG(history=100,nmixtures=10,backgroundRatio=0.9,noiseSigma=12),  "MOG Mask" ], # Gaussian Mixture background subtraction algorithm
    "mog2": [cv.createBackgroundSubtractorMOG2(history=20,varThreshold=50, detectShadows=True),        "MOG2 Mask"], # Gaussian Mixture background subtraction algorithm with support for shadows
    "knn":  [cv.createBackgroundSubtractorKNN(detectShadows=False,dist2Threshold=2000,history=100),         "KNN Mask" ], # K-nearest neighbors background subtraction algorithm
    "gsoc": [cv.bgsegm.createBackgroundSubtractorGSOC(), "GSOC Mask"], # Advanced background subtraction algorithm created during Google Summer of Code
    "cnt":  [cv.bgsegm.createBackgroundSubtractorCNT(),  "CNT Mask" ], # Counting based background subtraction algorithm (basic, but usable on inexpensive hardware)
    "gmg":  [cv.bgsegm.createBackgroundSubtractorGMG(),  "GMG Mask" ], # Algorithm developed for variable light condition tracking of human subjects
    "lsbp": [cv.bgsegm.createBackgroundSubtractorLSBP(), "LBSP Mask"]  # Local SVD Binary Pattern background subtraction algorithm
    } 
selection = "knn"
"""
So, which selections get us close to what we want?              "mog"  (Default already used) Least extraneous data, more disjoined results
                                                                "mog2" Shows shadows and clearly shows falling objects, but lots of extra results (slight temp variations) 
                                                                "knn"  Less extraneous temperature differences than "mog2" but still shows cluttered results
                                                                "cnt"  Disjointed results like MOG, still have a lot of extraneous results

And which selections are (with default parameters) not usable?  "lsbp" Absolutely unrecognizable
                                                                "gsoc" Absolutely unrecognizable 
                                                                "gmg"  Lags behind other algorithms, shows shadows better than objects

Overall, the best results are shown by the first three segmentation methods ("mog", "mog2", and "knn").
Next step is to refine the background segmentation parameters to improve recognition.
                                                                
"""


fgbg = bg_sub_method[selection][0]
title = bg_sub_method[selection][1]
print("Starting video comparison")
number_of_frames = 0
while True:
    ret, frame = cap.read()
    number_of_frames += 1
    if not ret:
        print("End of video found at frame ", number_of_frames, ".", sep="")
        break
    fgmask = fgbg.apply(frame,learningRate=-1)

    cv.imshow('Frame',frame) # Original video
    cv.imshow(title,fgmask)  # Video showing selected background segmentation method

    # Next section receives and handles user input (play/pause, quit commands supported)
    key = cv.waitKey(30)
    #print(key)
    if key in [ord('q'), 27]:
        print("Manual quit at frame ", number_of_frames, ".", sep="")
        break
    if key in [10,13,32]:
        print("Manually paused at frame ", number_of_frames, ". Press <Space> or <Enter> or <Return> to resume.", sep="")
        while cv.waitKey(30) not in [10,13,32]:
            "waiting"
        print("Resuming")

# Main loop exited due to end of file or user-initialized quit.
cap.release()
cv.destroyAllWindows()