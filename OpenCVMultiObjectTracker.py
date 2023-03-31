import numpy as np
import cv2 as cv
import math
from collections import deque
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment as lsa
from operator import attrgetter as getfield

MAX_UNSEEN_DURATION = 10   # Number of frames to wait before deleting unseen tracks
THRESHOLD = 128            # Max distance threshold (sum of squares) for track assignment
next_id = 1                # Global variable, stores ID of next (unassigned) track
tracks = []                # Global variable, holds all existing tracks

# Next two global variables are only used in internal testing - all instances of these can be removed from production
falling_detections = 0     # Global variable, overall number of object-frames flagged as potential fall object
tracks_culled = 0          # Global variable, overall number of expired tracks that were removed.


class KPCalc:
  """
  Note that the keypoint calculator used here is borrowed from Devin Bayly's work on Phase 1 of the project.

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
  Purpose: A track object represents an identified area of motion. Each track 
           object contains its own Kalman filter used for predicting frame-
           to-frame movement.
  
  Methods:
    __init__(self,id,box,centroid,parent=None)
      Creates a new Track object using the provided id, location/dimensions
      of box, location of centerpoint, and parent ID. Note that parent ID
      defaults to none.

      TODO: implement parent tracking to identify rockfall track sources.

    __correct(self,location)
      
  '''
  def __init__(self,id,box,centroid,parent=None):
    print(box)
    self.id = id
    self.bbox = box
    self.kalman_filter = KalmanFilter() # used for predicting individual track location
    self.age = 1                        # How many frames since the track first appeard
    self.total_visible = 1
    self.current_visible = 1
    self.unseen_duration = 0
    self.current_center = np.array([centroid[0],centroid[1]])
    self.previous_centers = deque(maxlen=5)
    self.previous_centers.append(self.current_center)
    self.bb_south_center = np.array([box[0]+(box[2]/2),box[1]+(box[3]/2)])
    self.leading_lower_edges = deque(maxlen=5)
    self.leading_lower_edges.append(self.bb_south_center)
    self.direction = 0
    self.magnitude = 0
    self.falling = False
    self.fall_duration = 0  # Time object is falling - used as alarm threshold.

  def correct(self,location):
    '''
    Purpose: Call track's individual Kalman filter's built-in correct() method.

    '''
    self.kalman_filter.correct(location)
    self.current_center = location
  
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
    #vect = self.vector(self.current_center, self.previous_centers[0])  # Uncomment this line and comment next for centroid-based motion characterization
    vect = self.vector(self.bb_south_center, self.leading_lower_edges[0])
    #print("Vector is",vect)
    magnitude = math.sqrt(vect[0]**2 + vect[1]**2)
    # normal = (vector[0] / magnitude, vector[1] / magnitude)
    direction = math.atan2(vect[1],vect[0]) * 180 / math.pi
    if direction < 0:
      direction = direction + 360
    fall = magnitude > 1 and 30 < direction < 150
    return (direction,magnitude,fall)


  def update(self,keypoint,visible,bounds=None):
    '''
    Updates all elements of a track, with the ability to handle seen or unseen
    tracks differently.
    '''
    global falling_detections
    if bounds != None:
      assert(keypoint != None)                                       # (bounds == None) == (keypoint == None)
      self.bbox = bounds                                             # Updates current bounding box to detected bounding box
      #print("Original prediction at",self.current_center,"but match found at",keypoint,"with a magnitude difference of",cv.norm(self.current_center, keypoint))
      self.correct(keypoint)                                         # Replaces predicted location with location of assigned detection
    else:
      #print("predicted locations:",self.kalman_filter.predicted_locations)
      locs = self.kalman_filter.predicted_locations
      prediction = self.current_center if len(locs) == 0 else locs[-1]
      shift = self.vector(prediction, self.current_center)
      self.correct(prediction)
      old_box = self.bbox
      self.bbox = (old_box[0] + shift[0], old_box[1] 
                   + shift[1], old_box[2], old_box[3])               # Shifts same bounding box to projected location for this frame
    
    self.age += 1                                                    # Number of frames since first seen

    #print("Previous center is",self.previous_centers[-1],"or",self.previous_centers[0],"and current center is",self.current_center,"but keypoint is",keypoint)

    self.previous_centers.append(self.current_center)                  # Maintains deque used for directionality estimate
    box = self.bbox
    self.leading_lower_edges.append(self.bb_south_center)
    self.bb_south_center = np.array([box[0]+(box[2]/2),box[1]+(box[3]/2)])
    #print("all centers:",self.previous_centers)

    if visible:                                                      # True if assigned (seen) this frame
      self.total_visible += 1
      self.current_visible += 1
      self.unseen_duration = 0
    else:                                                            # True if previously-assigned track has no assignment this frame
      self.current_visible = 0
      self.unseen_duration += 1

    #print("At update, track",id,"visibility is",visible,"and deque is",self.previous_centers)
    self.direction, self.magnitude, self.falling = self.find_direction()  # Calculates directionality and determines if object is falling
    
    # Following section monitors fall duration - not currently in use
    
    if self.falling:
      self.fall_duration += 1
      #print("FALL DETECTED!!!!!!!!!")
      falling_detections += 1
    else:
      self.fall_duration = 0
      #print("Not falling")

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
        #print("Point:",point)
        #print(type(point))
        new_point = np.array([[np.float32(point[0])],[np.float32(point[1])]])
        self.kalmanf.correct(new_point)
        self.place(point)

    def predict(self):
        '''
        Predicts tracked object's location in this frame (based on historical velocity).
        '''
        #print("Entered a kalman filter's predict() method.")
        tp = self.kalmanf.predict()
        self.predicted_locations.append((int(tp[0]),int(tp[1]))) 

    def place(self,new_location):
        '''
        Adds the object's assigned/identified location to the measured_locations array.
        '''
        self.measured_locations.append(new_location)

def retire():
  '''
  Removes tracks that have been without assignment for more than MAX_UNSEEN_DURATION frames.
  Prevents continued searching for motion that has stopped entirely.

  Should only be called if process_matches() finds at least one unassigned track.
  '''
  global tracks, tracks_culled
  orig_len = len(tracks)
  assert orig_len != 0, "Called retire() on zero-length tracks."
  tracks = [x for x in tracks if x.unseen_duration < MAX_UNSEEN_DURATION]
  new_len = len(tracks)
  if new_len != orig_len:
    tracks_culled += (orig_len - new_len)
    #print("one or more tracks expired: originally",orig_len,"tracks and now",new_len,"tracks.")

        
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
  #print("Predicts:",predicts,"\nDetects:",detects)
  predicts = pad(predicts,max_len*2)
  detects = pad(detects,max_len*2)
  #print("Predicts:",predicts,"\nDetects:",detects)
  cost_mat = distance.cdist(detects,predicts,'euclidean')
  cost_mat[0:len(detections),len(predictions):max_len*2] = threshold
  cost_mat[len(detections):max_len*2,0:len(predictions)] = threshold
  #print("matrix\n",cost_mat)
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
    #print("to pad:",to_pad,"\nadded:",added)
    to_pad = np.concatenate((to_pad,added))
  return to_pad

def optimize_cost_matrix(matrix):
  '''
  Purpose: Receives a 2-dimensional matrix of costs from the function call.
  Finds the optimized linear sum assignment using the Munkres Hungarian
  algorithm, and returns a pair of arrays representing the per-row optimized
  indices (always ordered) and the per-column optimized indices.
  '''
  #print(matrix)
  return lsa(matrix)

def process_matches(keypoints, boxes, assignments, cost_matrix):
  '''
  Purpose: Receives the lists of predicted locations, the solved
           assignments, and the cost matrix from the function call. For each
           entry in the assignments, determines if the assignment represents
           a needed track (caused by an identification beyond any track's
           predicted area) or an unmatched track (fewer detections than tracks
           in the frame). Calls retire() if at least one track is unseen.
  '''
  
  detections = [entry for entry in keypoints]
  predictions = [location.current_center for location in tracks]
  #print("Predictions:",predictions)
  cost_matrix = create_cost_matrix(predictions, detections, THRESHOLD)
  assignments = optimize_cost_matrix(cost_matrix)[1]

  # Need to create bounding boxes here *or* send to this function as an argument.
  
  # Categorize and handle assignments

  i = 0
  unseen_track = False
  while i < len(assignments):
    if i < len(detections):
      if assignments[i] < len(tracks):
        #print("Generating match for track", assignments[i], "at location", keypoints[i])
        tracks[assignments[i]].update(keypoints[i],True,bounds=boxes[i])
      else:
        make_track(boxes[i],keypoints[i])
    else:
      if assignments[i] < len(tracks):
        #print("Assignments:",assignments)
        #print("keypoints",keypoints)
        #print("len(tracks)",len(tracks))
        #print("len(keypoints)",len(keypoints))
        #print("line 272: i",i,"assignments[i]:",assignments[i],"len(tracks)",len(tracks),"len(keypoints)",len(keypoints),"len(detections)",len(detections))
        unseen_track = True
        tracks[assignments[i]].update(None,False)
    i += 1
  
  if unseen_track:      # At least 1 track without a matching detection this frame
    retire()      # Remove all lost tracks (unseen for too long) from tracks

def get_predictions():
  '''
  Generates numpy array of predictions for all active tracks, used in cost matrix for Munkres-Hungarian assignment
  '''
  #print("Entered get_predictions()")
  predictions = []
  for i in tracks:
    i.kalman_filter.predict()
    if len(i.kalman_filter.predicted_locations) > 0:
      predictions.append(i.kalman_filter.predicted_locations[-1])
  return predictions
   
def hungarian_assignments(centroids, boxes):
  '''
  Receives list of all detection centerpoints and bounding boxes for the frame, and associates with tracks
  based on Munkres-Hungarian assignment
  '''
  #print("Entered hungarian_assignments().")
  if len(centroids) > 0 and len(tracks) == 0: # No active tracks to match these detections
    detection = 0
    while detection < len(centroids):
      make_track(boxes[detection],centroids[detection])
      detection += 1
    return
  predictions = get_predictions()#[location.current_center for location in tracks]
  cost_matrix = create_cost_matrix(predictions, centroids, THRESHOLD)
  #print("Cost matrix is :", cost_matrix)
  assignments = optimize_cost_matrix(cost_matrix)
  #print("assignments",assignments)

  process_matches (centroids, boxes, assignments, cost_matrix)

def make_track(box, center):
  '''
  Creates a new empty track. Note that track to parent association is not yet implemented.
  '''
  #print("Entered make_track.")
  global next_id
  tracks.append(track(next_id,box,center,None))
  next_id += 1

def main():
  '''
  Main program loop. Not currently displaying tracks (still to implement)
  '''
  bg_sub_method = {
    "mog2": [cv.createBackgroundSubtractorMOG2(history=30,varThreshold=100,detectShadows=True),     "MOG2 Mask"], # Gaussian Mixture background subtraction algorithm with support for shadows
    "knn":  [cv.createBackgroundSubtractorKNN(detectShadows=False,dist2Threshold=2000,history=100), "KNN Mask" ], # K-nearest neighbors background subtraction algorithm
  } 

  selection = "knn"
  fgbg = bg_sub_method[selection][0]
  title = bg_sub_method[selection][1]
  print("Starting video comparison")
  number_of_frames = 0
  default_pixel = np.array([0,0,0])
  kpc = KPCalc()

  cap = cv.VideoCapture("C:\AppDev\BackgroundSegmentationTests\Examples\Loose debris drop with clear stable video.mp4")
  
  prev=[]

  while True:
    ret, frame = cap.read()
    number_of_frames += 1
    #print("Starting process of frame",number_of_frames)
    #print("number of tracks",len(tracks))
    if not ret:
      print("End of video found at frame ", number_of_frames, ".", sep="")
      print("Total object-frames flagged:", falling_detections)
      print("Total expired tracks:", tracks_culled)
      break
    fgmask = fgbg.apply(frame,learningRate=-1)
    fgmask = cv.morphologyEx(fgmask,cv.MORPH_CLOSE,np.ones((9,9),np.uint8))
    fgmask = cv.dilate(fgmask,np.ones((3,3),np.uint8),iterations = 1)
    fgmask = cv.morphologyEx(fgmask,cv.MORPH_OPEN,np.ones((1,2),np.uint8))
    frame_y,frame_x = fgmask.shape

    bg_image = fgbg.getBackgroundImage()

    frame2 = np.zeros((frame_y,frame_x,3),dtype=np.uint8)

    kpc.calcBlobs(fgmask)
    #if len(kpc.kp) > 0:
    #  print(kpc.kp[0].pt)
    #  print("centerpoints",[entry.pt for entry in kpc.kp]) # These are the centerpoints of detected blobs
    ## the following code helps to draw contours on an image 
    ret, thresh = cv.threshold(fgmask, 125,255,0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    ## draw the contour bound on the frame, this is in green but is simply the boundary of the contour
    cv.drawContours(frame2, contours, 0,(0,255,0),1)
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
        cv.rectangle(frame2, (bounds[0], bounds[1]),(bounds[0]+bounds[2],bounds[1]+bounds[3]),(255,0,0),1)
        prev.append(bounds)
        # print(bounds)
    if len(kpc.kp) > 0:
      hungarian_assignments([entry.pt for entry in kpc.kp], frame_boxes)
      #print(len(kpc.kp),"detections found")

    for entry in prev:
      cv.rectangle(frame2, (bounds[0], bounds[1]),(bounds[0]+bounds[2],bounds[1]+bounds[3]),(255,0,0),1)

    for track in tracks:
      box = track.bbox
      #print(box)
      color = (125,125,125) if track.current_visible == 0 else ((255,0,0) if track.falling else (0,255,255))
      cv.rectangle(frame, (int(box[0]), int(box[1])),(int(box[0]+box[2]),int(box[1]+box[3])),color,1)

    ## these are the steps for putting a circle at the blob's centroid coordinates
    blank = np.zeros((1, 1))
    empty = np.zeros(frame2.shape).astype("uint8")
    centroids = cv.drawKeypoints(empty, kpc.kp, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DEFAULT)
    # this essentially adds the image with the centroids drawn on top of the image frame we are processing

    frame2 += centroids

    cv.imshow('Frame',frame)                   # Original video
    cv.imshow(title,fgmask)                    # Foreground mask (segmented)
    cv.imshow('Background',bg_image)           # Background
    cv.imshow('Frame2',frame2)

    # Next section receives and handles user input
    key = cv.waitKey(5)
    if key in [ord('q'), 27]:
      print("Manual quit at frame ", number_of_frames, ".", sep="")
      break
    if key in [10,13,32]: # or len(kpc.kp) > 1:
      print("Manually paused at frame ", number_of_frames, ". Press <Space> or <Enter> or <Return> to resume.", sep="")
      while cv.waitKey(30) not in [10,13,32]:
        "waiting"
      print("Resuming")
    
  cap.release()  

# Main loop exited due to end of file or user-initialized quit.
main()

cv.destroyAllWindows()