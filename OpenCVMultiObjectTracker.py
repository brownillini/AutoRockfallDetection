import numpy as np
import cv2 as cv
import math
import os
from collections import deque
from scipy.spatial import distance
from scipy.optimize import linear_sum_assignment as lsa

MAX_UNSEEN_DURATION = 10   # Number of frames to wait before deleting unseen tracks
THRESHOLD = 40            # Max distance threshold (sum of squares) for track assignment
next_id = 1                # Global variable, stores ID of next (unassigned) track
tracks = []                # Global variable, holds all existing tracks

# Next two global variables are only used in internal testing - all instances of these can be removed from production
falling_detections = 0     # Global variable, overall number of object-frames flagged as potential fall object
tracks_culled = 0          # Global variable, overall number of expired tracks that were removed.

class track:
  '''
  Purpose: A track object represents an identified area of motion. Each track 
           object contains its own Kalman filter used for predicting frame-
           to-frame movement.

  Attributes:
    self: Required self-reference
    id: Track's unique id, will be used in parent tracking
    bbox: Bounding box of tracked object at latest frame
    kalman_filter: Track's Kalman Filter used to predict object location
    age: How many frames since the track first appeard
    total_visible: Total number of frames in which tracked object is visible
    current_visible: How long since the tracked object was unseen
    unseen_duration: How long since the tracked object was seen
    current_center: X/Y coordinate of the centroid
    previous_centers: List of last 5 centroid coordinates
    bb_south_center: X-center of bottom edge of bounding box
    leading_lower_edges: List of last 5 bb_south_center locations
    direction: Calculated travel direction of tracked object
    magnitude: Tracked object's speed of travel
    falling: Boolean representing characterized fall-like motion
    fall_duration: How long has the object been falling?
             
  Methods:
    __init__(self,id,box,centroid,parent=None)
      Creates a new Track object using the provided id, location/dimensions
      of box, location of centerpoint, and parent ID. Note that parent ID
      defaults to none.

      TODO: implement parent tracking to identify rockfall track sources.

    correct(self,location)
      Provides the tracked object's true location to the track's Kalman filter
      to update the predicted location.
  '''
  def __init__(self,id,box,centroid,parent=None):
    '''
    Parameters:
      self: Required self-reference
      id: Track's unique id, will be used in parent tracking
      box: Provided bounding box for tracked object
      centroid: Center point of tracked object
      parent: ID of previous object in a causal chain of falls
    '''
    self.id = id
    self.bbox = box
    self.kalman_filter = KalmanFilter()
    self.age = 1 
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
    self.parent = parent
    self.falling = False
    self.fall_duration = 0  # Time object is falling - used as alarm threshold.

  def correct(self,location):
    '''
    Purpose: Call track's individual Kalman filter's built-in correct() method.

    Parameters:
      self: Required self-reference
      location: X/Y coordinate of centroid
    '''
    self.kalman_filter.correct(location)
    self.current_center = location
  
  def vector(self, a, b):
    '''
    Purpose: Receives two 2-element arrays from the function call and returns
           a 2-element array representing the vector from the first element
           to the second.
    
    Parameters:
      self: Required self-reference
      a: First coordinate location
      b: Second coordinate location
    '''
    return [a[0] - b[0], a[1] - b[1]]
  
  def find_direction(self):
    '''
    Purpose: Determines overall direction, rate of movement, and potential fall
             based on historical location of track. Motion characterization is
             determined by tracking the lower center point of the bounding box
             (centroid-based motion is affected by upward exapansion, such as
             when a tracked object includes its rising debris cloud).

    Parameters:
      self: Required self-reference
    '''
    #vect = self.vector(self.current_center, self.previous_centers[0])  # Uncomment this line and comment next for centroid-based motion characterization
    vect = self.vector(self.bb_south_center, self.leading_lower_edges[-1])
    magnitude = math.sqrt(vect[0]**2 + vect[1]**2)
    direction = math.atan2(vect[1],vect[0]) * 180 / math.pi
    if direction < 0:
      direction = direction + 360
    fall = magnitude > 5 and 30 < direction < 150
    return (direction,magnitude,fall)


  def update(self,keypoint,visible,bounds=None):
    '''
    Purpose: Updates all elements of a track, with the ability to handle seen 
             or unseen tracks differently.

    Parameters:
      self: Required self-reference
      keypoint: Centroid of assigned detection
      visible: Boolean representing whether track is seen in this frame
      bounds: Bounding box of assigned detection    
    '''
    global falling_detections
    if bounds != None:
      assert(keypoint != None)                                       # (bounds == None) == (keypoint == None)
      self.bbox = bounds                                             # Updates current bounding box to detected bounding box
      self.correct(keypoint)                                         # Replaces predicted location with location of assigned detection
    else:
      locs = self.kalman_filter.predicted_locations
      prediction = self.current_center if len(locs) == 0 else locs[-1]
      shift = self.vector(prediction, self.current_center)
      self.correct(prediction)
      old_box = self.bbox
      self.bbox = (old_box[0] + shift[0], old_box[1] 
                   + shift[1], old_box[2], old_box[3])               # Shifts same bounding box to projected location for this frame
    
    self.age += 1                                                    # Number of frames since first seen

    self.previous_centers.append(self.current_center)                # Maintains deque used for directionality estimate
    box = self.bbox
    self.leading_lower_edges.append(self.bb_south_center)
    self.bb_south_center = np.array([box[0]+(box[2]/2),box[1]+(box[3]/2)])
    
    if visible:                                                      # True if assigned (seen) this frame
      self.total_visible += 1
      self.current_visible += 1
      self.unseen_duration = 0
    else:                                                            # True if previously-assigned track has no assignment this frame
      self.current_visible = 0
      self.unseen_duration += 1

    self.direction, self.magnitude, self.falling = self.find_direction()  # Calculates directionality and determines if object is falling
    
    # Following section monitors fall duration - not currently in use
    
    if self.falling:
      self.fall_duration += 1
      falling_detections += 1
    else:
      self.fall_duration = 0
    
class KalmanFilter:
    '''
    Purpose: Creates a single-object Kalman filter to track anticipated 
             movement. Each track will need an associated KalmanFilter.

    Attributes:
      predicted_locations: History of predicted locations for object
      measured_locations: History of actual locations for object
      kalman_filter: OpenCV KalmanFilter object
    '''

    def __init__(self):
        '''
        Parameters:
          self: Required self-reference
        '''
        q = 1e-5 # Q multiplier for processNoiseCov (the process noise covariace matrix)
        r = 1e-6 # R multiplier for measurementNoiseCov (the measurement noise covariance matrix)
        self.predicted_locations = []
        self.measured_locations = []

        # Set parameters for Kalman filter
        kalman = cv.KalmanFilter(4,2)
        kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * q
        kalman.measurementNoiseCov = np.array([[1,0],[0,1]],np.float32) * r
        self.kalman_filter = kalman
    
    def correct(self,point):
        '''
        Purpose: Corrects Kalman filter's location (replaces predicted location
                 with detected location, for more accurate prediction of next 
                 frame's location).

        Parameters:
          self: Required self-reference
          point: Location of tracked object, from Munkres/Hungarian assignment
        '''
        new_point = np.array([[np.float32(point[0])],[np.float32(point[1])]])
        self.kalman_filter.correct(new_point)
        self.place(point)

    def predict(self):
        '''
        Purpose: Predicts tracked object's location in this frame (based on 
                 historical velocity).
        
        Parameters:
          self: Required self-reference
        '''
        tp = self.kalman_filter.predict()
        self.predicted_locations.append((int(tp[0]),int(tp[1]))) 

    def place(self,new_location):
        '''
        Purpose: Adds the object's assigned/identified location to the 
                 measured_locations array.
        
        Parameters:
          self: Required self-reference
          new_location: X/Y coordinate identified for tracked object
        '''
        self.measured_locations.append(new_location)

def retire():
  '''
  Purpose: Removes tracks that have been without assignment for more than 
           MAX_UNSEEN_DURATION frames. Prevents continued searching for motion
           that has stopped entirely. Updates the global counter of how many
           tracks have been removed (tracks_culled)
           
           * Should only be called if process_matches() finds at least one 
           unassigned track.
  
  Global variables:
    tracks, tracks_culled
  '''
  global tracks, tracks_culled
  orig_len = len(tracks)
  assert orig_len != 0, "Called retire() on zero-length tracks."
  tracks = [x for x in tracks if x.unseen_duration < MAX_UNSEEN_DURATION]
  new_len = len(tracks)
  if new_len != orig_len: # At least one track removed this frame
    tracks_culled += (orig_len - new_len)
        
def create_cost_matrix(predictions, detections, threshold):
  '''
  Purpose: Receives a list of predicted locations (x/y tuples), a list of 
           foreground blob centroids (x/y tuples), and a distance threshold as 
           arguments. Uses this information to calculate an optimum cost matrix
           (minimum average distance between predictions and detections, with 
           any distances exceeding the threshold excluded from consideration). 
           Returns this matrix to the function call.

  Parameters:
    predictions: X/Y coordinate tuples representing kalman prediction per track
    detections: X/Y coordinate tuples representing detected blob centerpoints
    threshold: integer representing cost of nonassignment for the matrix

  Returns:
    cost_mat: the completed cost matrix
  '''
  max_len = max(len(predictions),len(detections))
  predicts, detects = np.array(predictions), np.array(detections)
  predicts = pad(predicts,max_len*2)
  detects = pad(detects,max_len*2)
  cost_mat = distance.cdist(detects,predicts,'euclidean')
  cost_mat[0:len(detections),len(predictions):max_len*2] = threshold
  cost_mat[len(detections):max_len*2,0:len(predictions)] = threshold
  return cost_mat    

def pad(to_pad,length):
  '''
  Purpose: Receives a list of tuples and a desired length from the function
           call. Extends the lists to the desired length by adding tuples of
           coordinates large enough to exceed reasonable match distances. Allows
           for creation of square, matchable cost matrices for use in scipy's 
           linear sum assignment.

  Parameters:
    to_pad: Original list of coordinate tuples
    length: Desired post-padding length

  Returns:
    to_pad: Original list, updated by adding extra (0,0) tuples to desired size
  '''
  current_length = len(to_pad)
  if to_pad.size == 0:
    to_pad = [(0,0)]
  if current_length < length:
    difference = length - current_length
    added = np.full(shape=(difference,2),fill_value=0,dtype=int)
    to_pad = np.concatenate((to_pad,added))
  return to_pad

def optimize_cost_matrix(matrix):
  '''
  Purpose: Receives a 2-dimensional matrix of costs from the function call.
           Finds an optimized linear sum assignment using the Munkres Hungarian
           algorithm, and returns a pair of arrays representing the per-row 
           optimized indices (always ordered) and the per-column optimized 
           indices.

  Parameter:
    matrix: cost matrix to optimize

  Returns:
    Array of arrays.
      Position [0] is the per-row optimization
      Position [1] is the per-column optimization
  '''
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
  cost_matrix = create_cost_matrix(predictions, detections, THRESHOLD)
  assignments = optimize_cost_matrix(cost_matrix)[1]
  
  # Categorize and handle assignments
  i = 0
  unseen_track = False
  while i < len(assignments):
    if i < len(detections):
      if assignments[i] < len(tracks): # Detection matched with existing track
        #print(i,len(boxes),len(tracks),len(assignments))
        tracks[assignments[i]].update(keypoints[i],True,bounds=boxes[i])
      else:                            # Detection needs new track
        make_track(boxes[i],keypoints[i])
    else:
      if assignments[i] < len(tracks): # No valid detection for this frame
        unseen_track = True
        tracks[assignments[i]].update(None,False)
    i += 1
  
  if unseen_track:  # At least 1 track without a matching detection this frame
    retire()        # Remove all lost tracks (unseen for too long) from tracks

def get_predictions():
  '''
  Generates numpy array of predictions for all active tracks, used in cost matrix for Munkres-Hungarian assignment
  '''
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
  if len(centroids) > 0 and len(tracks) == 0: # No active tracks to match these detections
    detection = 0
    #print(len(boxes), len(centroids),len(tracks))
    while detection < len(centroids):
      make_track(boxes[detection],centroids[detection])
      detection += 1
    return
  predictions = get_predictions()#[location.current_center for location in tracks]
  cost_matrix = create_cost_matrix(predictions, centroids, THRESHOLD)
  assignments = optimize_cost_matrix(cost_matrix)
  
  process_matches (centroids, boxes, assignments, cost_matrix)

def make_track(box, center):
  '''
  Creates a new empty track. Note that track to parent association is not yet implemented.
  '''
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
  number_of_frames = 0

  file_path = "C:\\AppDev\\BackgroundSegmentationTests\\Examples\\atrium.mp4"
  valid_file_path = os.path.exists(file_path)
  if not valid_file_path:
    print("Invalid input video path received. Exiting.")
    exit(1)
  print("Starting video comparison")
  cap = cv.VideoCapture(file_path)

  while cap.isOpened():
    ret, frame = cap.read()
    prev=[]
    number_of_frames += 1
    if not ret:
      print("End of video found at frame ", number_of_frames, ".", sep="")
      print("Total object-frames flagged:", falling_detections)
      print("Total expired tracks:", tracks_culled)
      print("Total tracks at end of video:", len(tracks))
      break
    fgmask = fgbg.apply(frame,learningRate=-1)
    fgmask = cv.morphologyEx(fgmask,cv.MORPH_CLOSE,np.ones((9,9),np.uint8))
    fgmask = cv.dilate(fgmask,np.ones((3,3),np.uint8),iterations = 1)
    fgmask = cv.morphologyEx(fgmask,cv.MORPH_OPEN,np.ones((1,2),np.uint8))
    frame_y,frame_x = fgmask.shape

    bg_image = fgbg.getBackgroundImage()

    frame2 = np.zeros((frame_y,frame_x,3),dtype=np.uint8)

    # the following code helps to draw contours on an image 
    ret, thresh = cv.threshold(fgmask, 125,255,0)
    contours,_ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # draw the contour bound on the frame, this is in green but is simply the boundary of the contour
    cv.drawContours(frame2, contours, 0,(0,255,0),1)
    # decide on the bounds of the contour to make into a box
    frame_boxes = [] # List of bounding boxes in frame (corresponds to centerpoints)
    centerpoints = []
    for i, c in enumerate(contours):
      # Calculate all moments up to 3rd order of the contour
      moments = cv.moments(c)
      # Calculate the mass center of the contour and add to centerpoints
      if moments["m00"] == 0: # cannot calculate centerpoint due to zero division
        print(c)
        print(moments["m10"],moments["m01"],moments["m00"])
        continue # Discard this data and skip to next contour
      centerpoints.append((int(moments["m10"]/moments["m00"]),int(moments["m01"]/moments["m00"])))
      # approximate the polygon without a high level of detail
      contour_poly = cv.approxPolyDP(c, 3, True)
      # create a bounding box for the contour (rectangle tightly enclosing the shape)
      bounds = cv.boundingRect(contour_poly)
      frame_boxes.append(bounds)
      if bounds[2] > 0 and bounds[3] > 0: # Alter this if we want to eliminate small detections
        # Draw the bounding box and center point on the frame
        cv.rectangle(frame2, (bounds[0], bounds[1]),(bounds[0]+bounds[2],bounds[1]+bounds[3]),(255,0,0))
        cv.circle(frame2,centerpoints[-1],3,(0,0,255),1)
        prev.append(bounds)
    assert len(frame_boxes) == len(centerpoints), "Mismatch between bounding box and centerpoint counts"
    if len(frame_boxes) > 0: # If true, at least one motion detection in this frame
      #print("Should be assigning matches at frame", number_of_frames)
      hungarian_assignments(centerpoints, frame_boxes)
       
    for entry in prev:
      cv.rectangle(frame2, (entry[0], entry[1]),(entry[0]+entry[2],entry[1]+entry[3]),(255,0,0),1)

    for track in tracks:
      box = track.bbox
      color = (125,125,125) if track.current_visible == 0 else ((255,0,0) if track.falling else (255,255,255))
      cv.rectangle(frame, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), color, 1)
      cv.putText(frame, str(track.id) + " " + str(track.total_visible), (int(box[0]), int(box[1])), 1, 1, color, 1, 1, False)

    cv.imshow('Frame',frame)                         # Original video
    cv.imshow('Segmented via ' + title,fgmask)       # Foreground mask (segmented)
    cv.imshow('Compared Background',bg_image)        # Background
    cv.imshow('Centroids and Bounding Boxes',frame2) # Centroid/bounds display

    # Next section receives and handles user input
    key = cv.waitKey(30) # Alter this to speed or slow processing
    if key in [ord('q'), 27]: # <Q> / <Escape>
      print("Manual quit at frame ", number_of_frames, ".", sep="")
      exit(0)
    if key in [10,13,32]: # <Space> / <Enter> / <Return>
      print("Manually paused at frame ", number_of_frames, ". Press <Space> or <Enter> or <Return> to resume.", sep="")
      while cv.waitKey(30) not in [10,13,27,32,ord('q')]:
        "waiting"
      print("Resuming")
    
  cap.release()  

  


main()

cv.destroyAllWindows() # Main loop exited due to end of file or user-initialized quit, still need to close windows.