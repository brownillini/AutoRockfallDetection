import numpy as np
import cv2 as cv

cap = cv.VideoCapture('test.mp4')
fgbg = cv.bgsegm.createBackgroundSubtractorGSOC()
print("Starting loop")
number_of_frames = 0
while True:
    ret, frame = cap.read()
    number_of_frames += 1
    if frame is None:
        print("End of video found at frame ", number_of_frames, ".")
        break
    fgmask = fgbg.apply(frame)

    cv.imshow('Frame',frame)
    cv.imshow('MOG Mask',fgmask)

    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

cap.release()
cap.destroyAllWindows()