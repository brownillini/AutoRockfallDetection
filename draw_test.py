import argparse
import cv2 as cv
import numpy as np

operation = 0
start = None
mouse_x = None
mouse_y = None
cursor_size = 10

def reset():
    return np.zeros((480,640,3))

def mouse_mask(event,x,y,flags,_):
    global operation, start, mouse_x, mouse_y, cursor_size

    mouse_x, mouse_y = x, y

    match flags:
        case 1:
            if start == None:
                start = (x,y)
            else:
                cv.line(background,(start[0],start[1]),(x,y),color=(255,255,255),thickness=cursor_size)
                start = (x,y)
        case 2:
            if start == None:
                start = (x,y)
            else:
                cv.line(background,(start[0],start[1]),(x,y),color=(0,0,0),thickness=cursor_size)
                start = (x,y)
        case 0:
            start = None
        case _:
            if flags > 100:
                cursor_size += 2
            elif flags < -100 and cursor_size > 2:
                cursor_size -= 2

arguments = argparse.ArgumentParser()
arguments.add_argument("-f", "--filename", help="Path to input file", default=".\\BuckeyeStep_Failure_Camera _20180705_043048 - Camera _1st_frame.png")
arguments.add_argument("-o", "--output_name", help="Path to output file", default=".\\BuckeyeStep_Failure_Camera _20180705_043048 - Camera mask2.png")

args = vars(arguments.parse_args())

background = reset()

cv.namedWindow("Mask")
cv.setMouseCallback("Mask",mouse_mask)

original = cv.imread(args["filename"])

while True:
    mask = background.copy()
    mask2 = original.copy()
    cv.circle(mask,(mouse_x,mouse_y),cursor_size // 2,(255,0,0),2) # This is the targeting circle, not part of the mask.
    #cv.imshow("Mask",mask)
    mask2[mask != (0,0,0)] = 255
    cv.imshow("Mask",mask2)

    key = cv.waitKey(1)
    if key in [ord('q')]:
        cv.destroyAllWindows()
        break
    elif key == ord('s'):
        cv.imwrite(args["output_name"],mask)
        cv.destroyAllWindows()
        break

cv.destroyAllWindows()