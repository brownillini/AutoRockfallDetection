import sys
import os
from os.path import isfile, isdir, join # helps keep find_all_videos concise
import cv2 as cv

RECOGNIZED_VIDEO_EXTENSIONS = ["mp4","avi","3gp"] # add more as needed

BG_SUB_METHODS = {
    "mog":  cv.bgsegm.createBackgroundSubtractorMOG(history=100,nmixtures=10,backgroundRatio=0.9,noiseSigma=12), # Gaussian Mixture background subtraction algorithm
    "mog2": cv.createBackgroundSubtractorMOG2(history=20,varThreshold=50, detectShadows=True),                   # Gaussian Mixture background subtraction algorithm with support for shadows
    "knn":  cv.createBackgroundSubtractorKNN(detectShadows=False,dist2Threshold=2000,history=100),               # K-nearest neighbors background subtraction algorithm
    "gsoc": cv.bgsegm.createBackgroundSubtractorGSOC(),                                                          # Advanced background subtraction algorithm created during Google Summer of Code
    "cnt":  cv.bgsegm.createBackgroundSubtractorCNT(),                                                           # Counting based background subtraction algorithm (basic, but usable on inexpensive hardware)
    "gmg":  cv.bgsegm.createBackgroundSubtractorGMG(),                                                           # Algorithm developed for variable light condition tracking of human subjects
    "lsbp": cv.bgsegm.createBackgroundSubtractorLSBP(),                                                          # Local SVD Binary Pattern background subtraction algorithm
    } 


def find_all_videos(path, method):
    print("Searching", path)
    for object in os.listdir(path):  # os.listdir() used as faster alternative to os.walk or glob.glob
        absolute_location = join(path, object)
        print(absolute_location)
        if isfile(absolute_location) and object.split(".")[-1] in RECOGNIZED_VIDEO_EXTENSIONS:
            process_bgsubtraction(absolute_location, method)
        elif isdir(absolute_location):
            find_all_videos(absolute_location, method)

def process_bgsubtraction(video, method):
    print("Processing video at",video)
    name, cap = get_source(video)                   # Input for video conversion
    output_name, output = initialize_output(name, cap)    # Output file
    fgbg = BG_SUB_METHODS[method]
    print("Starting video segmentation for ", name, ".", sep="")
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

def get_source(location):
    name = "".join(location.split(".")[0:-1])
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
    directory = os.getcwd()
    subtraction_method = "knn"
    if len(sys.argv) > 1: # True if command-line path provided.
        if isdir(sys.argv[1]):
            directory = sys.argv[1]
        else: 
            subtraction_method = sys.argv[1]
    if len(sys.argv) > 2 and sys.argv[2] in BG_SUB_METHODS.keys():
        subtraction_method = sys.argv[2]
    find_all_videos(directory,subtraction_method)

main()