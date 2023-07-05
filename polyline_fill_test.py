import math
import argparse
import cv2 as cv
import numpy as np

CLOSE_THRESHOLD = 100 # square of pixel distance to close polygon
MID_VALUE = 127       # color value corresponding to vertical drop

operation = 0
start = None
mouse_x = None
mouse_y = None
point_list = []
background = None
arrow_grid = None
overlay = None
output = None
mask = None
rotation_amount = 0
center_color = 255 // 2  # Integer, mid-gray
color_ratio = center_color / 90 # Float
rotation_window_size = 150
rotation = np.zeros((rotation_window_size,rotation_window_size,3))
centerpoint = (rotation_window_size // 2, rotation_window_size // 2)
window_x = 640
window_y = 480
window_depth = 3

def reset(x, y, depth):
    return np.zeros((x,y,depth),np.int16)

def reset_rotation():
    return np.zeros((rotation_window_size,rotation_window_size,3))

def update_rotation():
    global window_x, window_y
    arrow_length = int(len(rotation) * 0.4)
    show_rotation = rotation_amount * math.pi / 180
    flange_rotation1 = (rotation_amount + 90) * math.pi / 180
    flange_rotation2 = (rotation_amount - 90) * math.pi / 180
    arrowtip_x = int(math.sin(show_rotation) * arrow_length) + centerpoint[0]
    arrowtip_y = int(math.cos(show_rotation) * arrow_length) + centerpoint[1]
    lflange_x = int(math.sin(flange_rotation1) * arrow_length // 2) + centerpoint[0]
    lflange_y = int(math.cos(flange_rotation1) * arrow_length // 2) + centerpoint[1]
    rflange_x = int(math.sin(flange_rotation2) * arrow_length // 2) + centerpoint[0]
    rflange_y = int(math.cos(flange_rotation2) * arrow_length // 2) + centerpoint[1]
    cv.arrowedLine(rotation,centerpoint,(arrowtip_x,arrowtip_y),(255,255,255),3,tipLength=0.2)
    cv.line(rotation,centerpoint,(lflange_x,lflange_y),(255,255,255),3)
    cv.line(rotation,centerpoint,(rflange_x,rflange_y),(255,255,255),3)
    draw_arrows(window_x,window_y,show_rotation,40)

def draw_arrows(x,y,rotation,arrow_size):
    global arrow_grid
    arrow_grid = np.zeros_like(arrow_grid)
    mid = int(arrow_size / 2)
    y_start = 0
    while (y_start) < y:
        y_mid = y_start + mid
        x_start = 0
        while (x_start) < x:
            x_mid = x_start + mid
            end_x = int(math.sin(rotation) * mid) + x_mid
            end_y = int(math.cos(rotation) * mid) + y_mid
            start_x = -int(math.sin(rotation) * mid) + x_mid
            start_y = -int(math.cos(rotation) * mid) + y_mid
            cv.arrowedLine(arrow_grid,(start_x,start_y),(end_x,end_y),(0,255,0),8,tipLength=0.2)
            cv.arrowedLine(arrow_grid,(start_x,start_y),(end_x,end_y),(0,0,0),4, tipLength=0.2)
            x_start += arrow_size
        y_start += arrow_size

def draw_poly():
    global overlay, mask, arrow_grid, all_overlays, window_y,window_x,window_depth
    background = np.zeros((window_y,window_x,window_depth))
    cv.fillPoly(test,pts=np.array([point_list]),color=(rotation_value,rotation_value,rotation_value))
    cv.fillPoly(background,pts=np.array([point_list]),color=(0,0,0))
    cv.fillPoly(background,pts=np.array([point_list]),color=(255,255,255))
    overlay[background != (0,0,0)] = arrow_grid[background!=(0,0,0)]
    print("Should have a polygon here")
    clear_list()

def clear_list():
    global point_list
    point_list = []

def distance_squared(point_a, point_b):
    return (point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2

def mouse_rotate(event,x,y,flags,_):
    global rotation_amount

    mouse_x, mouse_y = x, y
    match flags:
        case 1:
            vector_1 = mouse_x - centerpoint[0]
            vector_2 = mouse_y - centerpoint[1]
            direction = math.atan2(vector_1,vector_2) * 180 / math.pi
            print(direction)
            rotation_amount = direction
    pass

def mouse_poly(event,x,y,flags,_):
    global operation, start, mouse_x, mouse_y, cursor_size, point_list

    mouse_x, mouse_y = x, y

    match flags:
        case 1:
            print(point_list)
            if len(point_list) > 2 and distance_squared(point_list[0],[x,y]) < CLOSE_THRESHOLD:
                print("about to draw")
                draw_poly()
                clear_list()
            else:
                print("About to add to the point list")
                print("background[0]:",len(background[0]),"background[1]:",len(background[1]))
                if x < 10:
                    x = 0
                elif x > len(background[0]) - 10:
                    x = len(background[0])
                if y < 10:
                    y = 0
                elif y > len(background) - 10:
                    y = len(background)
                new_point = [x,y]
                point_list.append(new_point)
                print("List after append:",point_list)


arguments = argparse.ArgumentParser()
arguments.add_argument("-f", "--filename", help="Path to input file", default=".\\BuckeyeStep_Failure_Camera _20180705_043048 - Camera _1st_frame.png")
arguments.add_argument("-o", "--output_name", help="Path to output file", default=".\\BuckeyeStep_Failure_Camera _20180705_043048 - Camera _angles.png")
arguments.add_argument("-m", "--mask", help="Path to mask output", default=".\\BuckeyeStep_Failure_Camera _20180705_043048 - Camera _mask.png")

args = vars(arguments.parse_args())

background = reset(window_y,window_x,window_depth)
result = background.copy()

cv.namedWindow("Fall Angle Overlay")
cv.setMouseCallback("Fall Angle Overlay",mouse_poly)
cv.namedWindow("Rotation")
cv.setMouseCallback("Rotation",mouse_rotate)

original = cv.imread(args["filename"])
print(original)
test = np.zeros_like(original)
test = cv.cvtColor(test,cv.COLOR_BGR2GRAY)
test = test + 127
test = cv.cvtColor(test,cv.COLOR_GRAY2BGR)
arrow_grid = np.zeros_like(test)
overlay = np.zeros_like(test)
blank = np.zeros_like(test)
all_overlays = np.zeros_like(test)

while True:
    mask = background.copy()
    mask2 = original.copy()
    

    point = 0
    while point < len(point_list):
        cv.circle(mask2,(point_list[point][0],point_list[point][1]),6,(0,255,0),4)
        if point > 1:
            cv.line(mask2,(point_list[point - 1][0],point_list[point - 1][1]),(point_list[point][0],point_list[point][1]),(255,255,0),2)
        point += 1
    update_rotation()
    rotation_value = int((rotation_amount + 180) * 255 / 360)
    mask2[overlay != (0,0,0)] = overlay[overlay!=(0,0,0)]
    blank = np.where(np.all(overlay != (0,0,0)), mask2, overlay)
    #cv.imshow("TRY THIS",blank)
    cv.imshow("Fall Angle Overlay",mask2)
    cv.imshow("Rotation",rotation)
    #cv.imshow("test",test)
    #cv.imshow("Overlay",overlay)
    key = cv.waitKey(1)
    if key != -1:
        print(key)
    if key in [ord('q')]:
        output = cv.cvtColor(test,cv.COLOR_BGR2GRAY)
        cv.imwrite(args["output_name"],test)
        break
    elif key == ord('c'):
        clear_list()
    elif key == ord('d') and len(point_list) > 2:
        draw_poly()
    elif key == 27: # Escape was pressed.
        clear_list()
    rotation = reset_rotation()

cv.destroyAllWindows()