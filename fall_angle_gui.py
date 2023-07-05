import PySimpleGUI as psg
import subprocess
import cv2 as cv
import numpy as np
import os

psg.theme("dark blue 14")

browse = [[psg.Text("Source Folder"), psg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
           psg.FolderBrowse()], [psg.Listbox(values=[], enable_events=True, size=(40, 20), key="-FILE LIST-")]]

result = [[psg.Text("File selected:")],[psg.Text(key="-FILE-")],[psg.Image(key="-VIDEO FRAME-")]]

layout = [[psg.Button("Mask"),psg.Button("Fall Angles"),psg.Button("Exit")],
          [psg.Button("Run")],
          [psg.Text("Select a source video for processing")],
          [psg.Column(browse),psg.VerticalSeparator(),psg.Column(result)]]

gui = psg.Window(title="Generate angle expectations and mask",layout=layout)

def first_frame_grab(gui,values):
    print("entered first_frame_grab()")
    try:
        filename = os.path.normpath(os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0]))
        no_extension = ".".join(filename.split(".")[0:-1])
        print(no_extension)
        _,frame = cv.VideoCapture(filename).read()
        print("got here")
        img_name = no_extension + "_1st_frame.png"
        print(img_name)
        cv.imwrite(img_name,frame)
        print("Should have a new image now")
        gui["-FILE-"].update(filename)
        gui["-VIDEO FRAME-"].update(filename=img_name)
        print(img_name)
        return frame, img_name, no_extension, filename
    except:
        raise Exception("File not found or video unreadable.")

def blank_frame(image):
    x = len(image[0])
    y = len(image)
    print("Width =",x,"and height =",y)
    frame = np.zeros((y,x,3))
    return frame

def run_process(arguments):
    subprocess.run(arguments)

def draw_mask(image_name,name_root):
    print("entered draw_mask")
    commands = ["python", ".\\draw_test.py", "-f", image_name, "-o", name_root + "_mask.png"]
    run_process(commands)

def map_fall_angles(image_name,name_root):
    commands = ["python", ".\\polyline_fill_test.py", "-f", image_name, "-o", name_root + "_angles.png"]
    run_process(commands)

def rockfall_tracker(file, mask, direction):
    commands = ["python", ".\\stabilized_tracker0621.py", "-f", file]
    if mask != "":
        commands += ["-m", mask]
    if direction != "":
        commands += ["-dm", direction]
    run_process(commands)

background = None
frame_generated = False
image_name = ""
name_root = ""
mask_name = ""
direction_name = ""
size = 50
while True:
    input,values = gui.read()
    if input == "Exit" or input == psg.WINDOW_CLOSED:
        break
    elif input == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            file_list = os.listdir(folder)
        except:
            file_list = []
        fnames = [file for file in file_list if os.path.isfile(os.path.join(folder, file)) and file.lower().endswith((".mp4", ".avi"))]
        gui["-FILE LIST-"].update(fnames)
    elif input == "-FILE LIST-":
        print("About to try framegrab")
        try:
            background,image_name,name_root,video_name = first_frame_grab(gui,values)
            frame_generated = True
        except:
            pass
    elif input == "Mask" and frame_generated is True:
        draw_mask(image_name,name_root)
        mask_name = name_root + "_mask.png"
    elif input == "Fall Angles" and frame_generated is True:
        map_fall_angles(image_name,name_root)
        direction_name = name_root + "_angles.png"
    elif input == "Run" and frame_generated is True:
        rockfall_tracker(video_name, mask_name, direction_name)

gui.close()
#blank_frame(background)