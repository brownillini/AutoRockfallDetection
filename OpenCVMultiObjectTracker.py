import numpy as np
import cv2 as cv
import math
#import pybgs as bgs
from collections import deque
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment as lsa
from operator import attrgetter as getfield

MAX_UNSEEN_DURATION = 15   # Number of frames to wait before deleting unseen tracks
THRESHOLD = 512            # Max distance threshold (sum of squares) for track assignment
next_id = 1                # Global variable, stores ID of next (unassigned) track
tracks = []                # Global variable, holds all existing tracks


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

class track:
  '''
  A track object represents an identified area of motion. Each track object contains its own Kalman filter used for predicting frame-to-frame movement.
  '''
  def __init__(self,id,box,centroid,parent=None):
    self.id = id
    self.bbox = box
    self.kalmanFilter = KalmanFilter() # used for predicting individual track location
    self.age = 1                       # How many frames since the track first appeard
    self.totalVisibleCount = 1
    self.visibleFor = 1
    self.consecutiveInvisibleCount = 0
    self.currentCenter = np.array([centroid[0],centroid[1]])
    self.previousCenters = deque(maxlen=5)
    self.previousCenters.append(self.currentCenter)
    self.direction = 0
    self.magnitude = 0
    self.falling = False
    self.fallDuration = 0  # Time object is falling - used as alarm threshold.

  def correct(self,location):
    '''
    Calls track's individual Kalman filter's built-in correct() method.
    '''
    self.kalmanFilter.correct(location)
    pass
  
  '''
  Purpose: Receives two 2-element arrays from the function call and returns
            a 2-element array representing the vector from the first element
            to the second.
  '''
  def vector(self, a, b):
    '''
    Determines x/y vector between two points
    '''
    return [a[0] - b[0], a[1] - b[1]]
  
  def find_direction(self):
    '''
    Determines overall direction, rate of movement, and potential fall based on
    historical location of track.
    '''
    vect = self.vector(self.currentCenter, self.previousCenters[-1])
    magnitude = math.sqrt(vect[0]**2 + vect[1]**2)
    # normal = (vector[0] / magnitude, vector[1] / magnitude)
    direction = math.atan2(vect[1],vect[0]) * 180 / math.pi
    if direction < 0:
      direction = direction + 360
    fall = magnitude > 1 and direction < 150 and direction > 30
    return (direction,magnitude,fall)


  def update(self,keypoint,visible,bounds=None):
    '''
    Updates all elements of a track, with the ability to handle seen or unseen
    tracks differently.
    '''
    if bounds != None:
      self.bbox = bounds                                                # Updates current bounding box to detected bounding box
    else:
      print(self.bbox)
      old_box = self.bbox
      shift = self.vector(keypoint, self.currentCenter)
      self.bbox = (old_box[0] + shift[0], old_box[1] 
                   + shift[1], old_box[2], old_box[3])   # Shifts same bounding box to projected location for this frame
    
    self.age += 1                                                    # Number of frames since first seen

    self.correct(self.currentCenter)                                # Replaces predicted location with location of assigned detection

    self.previousCenters.append(self.currentCenter)                    # Maintains deque used for directionality estimate
    
    if visible:                                                      # True if assigned (seen) this frame
      self.totalVisibleCount += 1
      self.visibleFor += 1
      self.consecutiveInvisibleCount = 0
    else:                                                            # True if previously-assigned track has no assignment this frame
      self.visibleFor = 0
      self.consecutiveInvisibleCount += 1

    self.direction, self.magnitude, self.falling = self.find_direction()  # Calculates directionality and determines if object is falling
    
    # Following section monitors fall duration - not currently in use
    
    if self.falling:
      self.fall_duration += 1
      print("FALL DETECTED!!!!!!!!!")
    else:
      self.fall_duration = 0
      print("Not falling")
    
    

class KalmanFilter:
    '''
    Creates a single-object Kalman filter to track anticipated movement. Each track will need an associated KalmanFilter.
    '''

    def __init__(self):
        q = 1e-5 # Q multiplier for processNoiseCov (the process noise covariace matrix)
        r = 1e-6 # R multiplier for measurementNoiseCov (the measurement noise covariance matrix)
        self.predicted_locations = []
        self.measured_locations = []
        kalman = cv.KalmanFilter(4,2)
        kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * q
        kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * r
        self.kalmanf = kalman
    
    def correct(self,point):
        '''
        Corrects Kalman filter's location (replaces predicted location with detected location, for
        more accurate prediction of next frame's location).
        '''
        print("Point:",point)
        print(type(point))
        new_point = np.array([[np.float32(point[0])],[np.float32(point[1])]])
        self.kalmanf.correct(new_point)

    def predict(self):
        '''
        Predicts tracked object's location in this frame (based on historical velocity).
        '''
        tp = self.kalmanf.predict()
        self.predicted_locations.append((int(tp[0]),int(tp[1]))) 

    def place(self,new_location):
        '''
        Adds the object's assigned/identified location to the measured_locations array.
        '''
        self.measured_locations.append(new_location)

class assignments():
  '''
  Munkres version of the Hungarian algorithm, includes a cost of non-assignment used to
  determine whether each track should be assigned or unassigned for that frame, as well
  as whether each detection will have an associated track or requires track creation.
  '''

  def __init__(self,tracks,detections):
    # alternate form of cost matrix using cost_matrix.py
    self.cost_matrix = create_cost_matrix(tracks, detections, THRESHOLD)

    # Take detection/track pairs and calculate euclidian distance for each pair.
    optimized = optimize_cost_matrix(self.cost_matrix)

def retire(tracks):
  '''
  Purpose: Removes any tracks that have not been seen for longer than
  MAX_UNSEEN_DURATION.
  '''
  if len(tracks) == 0: # No tracks at all
    return
  tracks = [x for x in tracks if x.consecutiveInvisibleCount < MAX_UNSEEN_DURATION]
        
def create_cost_matrix(predictions, detections, threshold):
  '''
  Purpose: Receives a list of predicted locations (x/y tuples), a list of 
  foreground blob centroids (x/y tuples), and a distance threshold as 
  arguments. Uses this information to calculate an optimum cost matrix 
  (minimum average distance between predictions and detections, with any
  distances exceeding the threshold excluded from consideration). Returns
  this matrix to the function call.
  '''
  max_len = max(len(predictions),len(detections))
  predicts, detects = np.array(predictions), np.array(detections)
  print("Predicts:",predicts,"\nDetects:",detects)
  predicts = pad(predicts,max_len*2)
  detects = pad(detects,max_len*2)
  print("Predicts:",predicts,"\nDetects:",detects)
  cost_mat = distance.cdist(detects,predicts,'euclidean')
  cost_mat[0:len(detections),len(predictions):max_len*2] = threshold
  cost_mat[len(detections):max_len*2,0:len(predictions)] = threshold
  print("matrix\n",cost_mat)
  return cost_mat    

def pad(to_pad,length):
  '''
  Purpose: Receives a list of lists and a desired list length from the 
  function call. Extends the lists to the desired length by adding tuples of
  coordinates large enough to exceed reasonable match distances. Allows for
  for creation of square, matchable cost matrices for use in scipy's linear
  sum assignment.
  '''
  current_length = len(to_pad)
  if to_pad.size == 0:
    to_pad = [(0,0)]
  if current_length < length:
    difference = length - current_length
    added = np.full(shape=(difference,2),fill_value=0,dtype=int)
    print("to pad:",to_pad,"\nadded:",added)
    to_pad = np.concatenate((to_pad,added))
  return to_pad

def optimize_cost_matrix(matrix):
  '''
  Purpose: Receives a 2-dimensional matrix of costs from the function call.
  Finds the optimized linear sum assignment using the Munkres Hungarian
  algorithm, and returns a pair of arrays representing the per-row optimized
  indices (always ordered) and the per-column optimized indices.
  '''
  print(matrix)
  return lsa(matrix)

def process_matches(keypoints, boxes, assignments, cost_matrix):
  '''
  Purpose: Receives the lists of predicted locations, the solved
           assignments, and the cost matrix from the function call. For each
           entry in the assignments, determines if the assignment represents
           a needed track (caused by an identification beyond any track's
           predicted area) or an unmatched track (fewer detections than tracks
           in the frame). Calls update_found(), update_orphan(), remove_lost()
           as necessary to handle each situation.
  '''
  
  detections = [entry for entry in keypoints]
  predictions = [location.currentCenter for location in tracks]
  print("Predictions:",predictions)
  cost_matrix = create_cost_matrix(predictions, detections, THRESHOLD)
  assignments = optimize_cost_matrix(cost_matrix)[1]

  # Need to create bounding boxes here *or* send to this function as an argument.
  
  # Categorize and handle assignments

  i = 0
  unseen_track = False
  while i < len(assignments):
    if i < len(detections):
      if assignments[i] < len(tracks):
        print("Generating match for track", assignments[i], "at location", keypoints[i])
        tracks[assignments[i]].update(keypoints[i],True,bounds=boxes[i])
      else:
        tracks.append(track(next_id,boxes[i],keypoints[i],None))
    else:
      if assignments[i] < len(tracks):
        print("line 272: assignments[i]:",i,"len(tracks)",len(tracks),"len(keypoints)",len(keypoints),"len(detections)",len(detections))
        unseen_track = True
        tracks[assignments[i]].update(keypoints[assignments[i]],False)
    i += 1
  
  if unseen_track:      # At least 1 track without a matching detection this frame
    retire(tracks)      # Remove all lost tracks (unseen for too long) from tracks

def update_found(assigned):
  '''
  Updates the location and stats of any track with an assigned identification this frame.
  '''
  for assigned_track in assigned:
    assigned_track.correct()
       

def update_orphan(unassigned):
  '''
  Updates the predicted location and stats of any track not seen during this frame.
  '''
  unassigned.age += 1
  unassigned.consecutiveInvisibleCount += 1

  pass

def remove_lost(tracking):
  '''
  Removes tracks that have been without assignment for more than MAX_UNSEEN_DURATION frames.
  Prevents continued searching for motion that has stopped entirely.

  Should only be called if process_matches() finds at least one unassigned track.
  '''
  remaining = [entry for entry in tracking if entry.unseen_for < MAX_UNSEEN_DURATION]
  tracking = remaining # updates pointer to shortened list, allows old list to be gc'd

def predictNewLocationsOfTracks():
  '''
  Updates predicted location for each track in this frame based on Kalman filter
  '''
  for track in tracks:
    bbox = track.bbox
    track.kalmanFilter.predict()
    predictedCenter = track.kalmanFilter.predicted_locations[-1]
    track.bbox = (predictedCenter[0]-bbox[2]//2, predictedCenter[1]-bbox[3]//2, bbox[2], bbox[3])

def getPredictions():
  '''
  Generates numpy array of predictions for all active tracks, used in cost matrix for Munkres-Hungarian assignment
  '''
  predictions = []
  for i in tracks:
    i.kalman_filter.predict()
    if len(i.kalman_filter.predicted_locations) > 0:
      predictions.append(i.kalman_filter.predicted_locations[-1])
  return predictions
   
def detectionToTrackAssignment(centroids, boxes):
  '''
  Receives list of all detection centerpoints and bounding boxes for the frame, and associates with tracks
  based on Munkres-Hungarian assignment
  '''
  print("Entered detectionToTrackAssignment().")
  if len(centroids) > 0 and len(tracks) == 0: # No active tracks to match these detections
    detection = 0
    while detection < len(centroids):
      make_track(boxes[detection],centroids[detection])
      detection += 1
    return
  predictions = [location.currentCenter for location in tracks]
  cost_matrix = create_cost_matrix(predictions, centroids, THRESHOLD)
  print("Cost matrix is :", cost_matrix)
  assignments = optimize_cost_matrix(cost_matrix)
  print("assignments",assignments)

  process_matches (centroids, boxes, assignments, cost_matrix)

def make_track(box, center):
  '''
  Creates a new empty track. Note that track to parent association is not yet implemented.
  '''
  print("Entered make_track.")
  global next_id
  tracks.append(track(next_id,box,center,None))
  next_id += 1

def main():
  '''
  Main program loop. Not currently displaying tracks (still to implement)
  '''
  bg_sub_method = {
    "mog2": [cv.createBackgroundSubtractorMOG2(history=30,varThreshold=100,detectShadows=True),        "MOG2 Mask"], # Gaussian Mixture background subtraction algorithm with support for shadows
    "knn":  [cv.createBackgroundSubtractorKNN(detectShadows=False,dist2Threshold=2000,history=100),         "KNN Mask" ], # K-nearest neighbors background subtraction algorithm
  } 

  selection = "knn"
  fgbg = bg_sub_method[selection][0]
  title = bg_sub_method[selection][1]
  print("Starting video comparison")
  number_of_frames = 0
  default_pixel = np.array([0,0,0])
  kpc = KPCalc()

  cap = cv.VideoCapture("C:\AppDev\BackgroundSegmentationTests\Examples\BouncingBallExample.mp4")
  
  prev=[]

  #seg_params = get_seg_params(selection)
  while True:
    ret, frame = cap.read()
    number_of_frames += 1
    if not ret:
      print("End of video found at frame ", number_of_frames, ".", sep="")
      break
    fgmask = fgbg.apply(frame,learningRate=-1)
    fgmask = cv.morphologyEx(fgmask,cv.MORPH_CLOSE,np.ones((9,9),np.uint8))
    fgmask = cv.dilate(fgmask,np.ones((3,3),np.uint8),iterations = 1)
    fgmask = cv.morphologyEx(fgmask,cv.MORPH_OPEN,np.ones((1,2),np.uint8))
    frame_y,frame_x = fgmask.shape

    bg_image = fgbg.getBackgroundImage()

    frame = np.zeros((frame_y,frame_x,3),dtype=np.uint8)

    kpc.calcBlobs(fgmask)
    if len(kpc.kp) > 0:
      #print(kpc.kp[0].pt)
      print([entry.pt for entry in kpc.kp]) # These are the centerpoints of detected blobs
    ## the following code helps to draw contours on an image 
    ret, thresh = cv.threshold(fgmask, 125,255,0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    ## draw the contour bound on the frame, this is in green but is simply the boundary of the contour
    cv.drawContours(frame, contours, 0,(0,255,0),1)
    ## decide on the bounds of the contour to make into a box
    frame_boxes = [] # List of bounding boxes in frame (corresponds to centerpoints)
    for i, c in enumerate(contours):
      # approximate the polygon without a high level of detail
      contour_poly = cv.approxPolyDP(c, 3, True)
      #figure out the min and max x,y positions in the contour polygon
      bounds = cv.boundingRect(contour_poly)
      frame_boxes.append(bounds)
      # calculate the area
      if bounds[2] > 1 and bounds[3] > 1:
        #draw a rectangle on the frame using the corners of the box. this will be in blue
        cv.rectangle(frame, (bounds[0], bounds[1]),(bounds[0]+bounds[2],bounds[1]+bounds[3]),(255,0,0),1)
        prev.append(bounds)
    if len(kpc.kp) > 0:

      detectionToTrackAssignment([entry.pt for entry in kpc.kp], frame_boxes)
      print("detections found")

    for entry in prev:
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

    # Next section receives and handles user input
    key = cv.waitKey(30)
    if key in [ord('q'), 27]:
      print("Manual quit at frame ", number_of_frames, ".", sep="")
      break
    if key in [10,13,32]:
      print("Manually paused at frame ", number_of_frames, ". Press <Space> or <Enter> or <Return> to resume.", sep="")
      while cv.waitKey(30) not in [10,13,32]:
        "waiting"
      print("Resuming")
  cap.release()  

# Main loop exited due to end of file or user-initialized quit.
main()

cv.destroyAllWindows()