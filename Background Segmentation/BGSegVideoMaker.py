import sys
import numpy as np
import cv2 as cv

def get_source(location):
    name = location.split(".")[0]                   # Currently expects only one period in filename
    cap = cv.VideoCapture(location)
    return name, cap

def initialize_output(name, cap):
    output_name = name + "_segmented-BG.avi"        # Output in .avi only (simplifies codec selection)
    codec = cv.VideoWriter_fourcc(*'MJPG')          # Codec for .avi output
    frame_rate = cap.get(5)                         # Input file's framerate (float)
    resolution = (int(cap.get(3)), int(cap.get(4))) # Width / Height of input file

    # The "False" boolean at the end indicates that the output is a B&W video file.
    output = cv.VideoWriter(output_name, codec, frame_rate, resolution, False)

    return output_name, output

def main():
    if len(sys.argv) == 1: # True if command-line execution did not include conversion target.
        print("Missing required parameter: Original filename.")
        print(f"Example: {sys.argv[0]} <test.mp4>")
        exit(1)
    name, cap = get_source(sys.argv[1])                   # Input for video conversion
    output_name, output = initialize_output(name, cap)    # Output file

    # Choose a background subtraction algorithm (these are all of OpenCV's included background subtractors).

    bg_sub_method = {
        "mog":  cv.bgsegm.createBackgroundSubtractorMOG(history=100,nmixtures=10,backgroundRatio=0.9,noiseSigma=12), # Gaussian Mixture background subtraction algorithm
        "mog2": cv.createBackgroundSubtractorMOG2(history=20,varThreshold=50, detectShadows=True),                   # Gaussian Mixture background subtraction algorithm with support for shadows
        "knn":  cv.createBackgroundSubtractorKNN(detectShadows=False,dist2Threshold=2000,history=100),               # K-nearest neighbors background subtraction algorithm
        "gsoc": cv.bgsegm.createBackgroundSubtractorGSOC(),                                                          # Advanced background subtraction algorithm created during Google Summer of Code
        "cnt":  cv.bgsegm.createBackgroundSubtractorCNT(),                                                           # Counting based background subtraction algorithm (basic, but usable on inexpensive hardware)
        "gmg":  cv.bgsegm.createBackgroundSubtractorGMG(),                                                           # Algorithm developed for variable light condition tracking of human subjects
        "lsbp": cv.bgsegm.createBackgroundSubtractorLSBP(),                                                          # Local SVD Binary Pattern background subtraction algorithm
        } 
    selection = "knn"

    fgbg = bg_sub_method[selection]
    print("Starting video segmentation")
    number_of_frames = 0

    # Following loop segments each frame from original and saves result to output video.
    while True:
        ret, frame = cap.read()
        number_of_frames += 1
        if not ret: # True if end of file reached
            print("End of video found at frame ", number_of_frames, ".", sep="")
            print("Background segmentation saved as ", output_name, ".", sep="")
            break
        fgmask = fgbg.apply(frame,learningRate=-1)
        output.write(fgmask)

    # Close opened files (original and new background-segmented file)
    cap.release()
    output.release()

main()