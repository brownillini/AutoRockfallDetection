import numpy as np
import cv2 as cv
#import pybgs as bgs

class KPCalc:
  """
  A class representing the keypoint calculator. This is using the SimpleBlobDetector from OpenCV.
  The parameters used here are from the default listed in https://learnopencv.com/blob-detection-using-opencv-python-c/

  Attributes:

  kp: list
     this is a list of the keypoints that we get back when we run the detector.detect(im) line
  detector: SimpleBlobDetector
     this is a opencv blob detector built from the parameters listed
  params: SimpleBlobDetector_Params
     the parameters that control whether a blob in the image is considered a blob to get the keypoint of.

  Methods:

  init(self)
    this just generates the blob detector from the parameters and sets the instance attribute variables
  calcBlobs(self,im)
    the image is a black and white frame from the video that we get after the background segmentation has completed
    when the self.detector is run we get back a list of keypoint objects. They are defined by https://docs.opencv.org/3.4/d2/d29/classcv_1_1KeyPoint.html
    and we can access their pt property to get back x,y coordinates in screen space for what the centroid of the blob was.
  """

  def __init__(self):
    params = cv.SimpleBlobDetector_Params()
    params.minThreshold = 1
    params.maxThreshold = 20
    params.minArea = 5
    params.maxArea = 500
    params.filterByInertia = False
    params.filterByConvexity = False
    params.blobColor = 255
    detector = cv.SimpleBlobDetector_create(params)
    detector.empty()

    self.kp = []
    self.params = params
    self.detector = detector

  def calcBlobs(self, im):
    self.kp = self.detector.detect(im)
    return self.kp

cap = cv.VideoCapture('./test.mp4')

bg_sub_method = {
    "mog2": [cv.createBackgroundSubtractorMOG2(history=30,varThreshold=100,detectShadows=True),        "MOG2 Mask"], # Gaussian Mixture background subtraction algorithm with support for shadows
    "knn":  [cv.createBackgroundSubtractorKNN(history=15,detectShadows=False,dist2Threshold=2000),         "KNN Mask" ], # K-nearest neighbors background subtraction algorithm
    } 

selection = "knn"

'''
Following sections include display of three separate views: The original frame, the segmented mask,
and the background being considered at this frame. Mote that despite relatively little difference 
between background and current frame, there are still many noisy points of difference. 

Ideal comparison would involve running the background segmentation, then comparing frame pixels
corresponding to foreground mask detections with their background counterparts, and dropping the
foreground mask pixel (set to black) if these points are within a particular pixel threshold.

This mimics the thresholding that the above algorithms are already applying - but a postprocessing
step may reduce the detection noise to a better level.


'''

def get_seg_params(method):
  return bg_sub_method[method]

def main():
  fgbg = bg_sub_method[selection][0]
  title = bg_sub_method[selection][1]
  print("Starting video comparison")
  fgbg.setHistory(50)
  print("History value of model is", fgbg.getHistory())
  print("Nearest neighbors considered:",fgbg.getkNNSamples())
  number_of_frames = 0
  default_pixel = np.array([0,0,0])
  kpc = KPCalc()

  seg_params = get_seg_params(selection)
  while True:
    ret, frame = cap.read()
    number_of_frames += 1
    if not ret:
        print("End of video found at frame ", number_of_frames, ".", sep="")
        break
    fgmask = fgbg.apply(frame,learningRate=-1)
    #print("frame length is ", frame.size, "with shape", frame.shape)
    frame_y,frame_x = fgmask.shape
    #print("Frame x width is",frame_x,"and frame y height is",frame_y)

    #print("mask length is  ", fgmask.size, "with shape", fgmask.shape)

    bg_image = fgbg.getBackgroundImage()

    bg_filtered = np.zeros((frame_y,frame_x,3),dtype=np.uint8)
    #print("bg_filtered has size",bg_filtered.size,"and shape",bg_filtered.shape)

    x = 0
    
    '''
    # Replace following loop with matrix operations

    while x < frame_x:
        y = 0
        while y < frame_y:
            bgi = background_image[y][x]
            frm = frame[y][x]
            if fgmask[y][x] == 255: # and (abs(bgi[0]-frm[0]) < 100):
                #print("close red channel at ",x," ",y,".",sep="")
                bg_filtered[y][x] = [0,120,120]
            y+=1
        x+=1
    #filtered_image = im.fromarray(bg_filtered)
    '''
    
    #print(background_filtered)

    frame = np.zeros((frame_y,frame_x,3),dtype=np.uint8)


    kpc.calcBlobs(fgmask)
    ## the following code helps to draw contours on an image 
    ret, thresh = cv.threshold(fgmask, 125,255,0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    ## draw the contour bound on the frame, this is in green but is simply the boundary of the contour
    cv.drawContours(frame, contours, 0,(0,255,0),1)
    ## decide on the bounds of the contour to make into a box
    bounding = []
    for i, c in enumerate(contours):
      # approximate the polygon without a high level of detail
      contour_poly = cv.approxPolyDP(c, 3, True)
      #figure out the min and max x,y positions in the contour polygon
      bounds = cv.boundingRect(contour_poly)
      # calculate the area
      if bounds[2] > 1 and bounds[3] > 1:
        #draw a rectangle on the frame using the corners of the box. this will be in blue
        cv.rectangle(frame, (bounds[0], bounds[1]),(bounds[0]+bounds[2],bounds[1]+bounds[3]),(255,0,0),1)

    ## these are the steps for putting a circle at the blob's centroid coordinates
    blank = np.zeros((1, 1))
    empty = np.zeros(frame.shape).astype("uint8")
    centroids = cv.drawKeypoints(empty, kpc.kp, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DEFAULT)
    # this essentially adds the image with the centroids drawn on top of the image frame we are processing

    frame += centroids

    cv.imshow('Frame',frame)                   # Original video
    cv.imshow(title,fgmask)                    # Foreground mask (segmented)
    cv.imshow('Background',bg_image)           # Background
    #cv.imshow('Filtered', bg_filtered)         # Filtered image

    # Next section receives and handles user input
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
    #print("Next frame!")

# Main loop exited due to end of file or user-initialized quit.
main()
cap.release()
cv.destroyAllWindows()