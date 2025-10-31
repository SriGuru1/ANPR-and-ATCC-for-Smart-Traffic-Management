#taffic monitoring and license plate detection
# 1. process the video of traffic
# 2. detect traffic light whether its red, yellow , green 
# 3. lane marking(white marks) to track vehicle movement
# 4. vehicle license plate is detected using the haar cascade detection 
# 5. OCR -> Extracting the character from the license plate
# 6. logs violation -> mysql database
# 7. Display the processed video with annotations

import cv2
import numpy as np 
import matplotlib.pyplot as plt
import os
import mysql.connector
import re
from PIL import Image
from collections import deque 
import pytesseract  # OCR (OPTICAL CHARACTER RECOGNISTION) 
import easyocr
from mysql.connector import Error

DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = 'BABLUinfosys@12345'
DB_NAME = 'mydb'



#Define the license plate cascade 
license_plate_cascade = cv2.CascadeClassifier(
    r"C:\Users\Admin\ANPR-and-ATCC-for-Smart-Traffic-Management\models\YOLO\haarcascade_russian_plate_number.xml"
)

# Haar cascade classifier - helps in object detection used in cv.
#ensure the file exists and is loaded correctly
if license_plate_cascade.empty():
    raise FileNotFoundError("Haar Cascade for license plate detection not found. Ensure the path is correct.")

def detect_traffic_color(image, rect):
    x,y,w,h  = rect
    # Extract region of Interest (ROI) from the image based on the rectangle

    roi = image[y:y+h,x:x+w]

    #convert ROI to HSV color space #HSV -hue saturation Value

    hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    
    #Define HSV range for red color
    red_lower = np.array([0,120,70])
    red_upper = np.array([10,255,255])

    #Define HSV range for yellow color
    yellow_lower = np.array([20,100,100])
    yellow_upper = np.array([30,255,255])

    red_mask = cv2.inRange(hsv,red_lower,red_upper)
    yellow_mask = cv2.inRange(hsv,yellow_lower,yellow_upper)
    font, font_scale, font_thickness = cv2.FONT_HERSHEY_TRIPLEX, 1, 2
    if cv2.countNonZero(red_mask) > 0:
        text_color, message, color = (0,0,255), "Detected Signal Status : Stop", 'red'
    elif cv2.countNonZero(yellow_mask) > 0:
        text_color, message, color = (0,255,255), "Detected Signal Status: Caution", 'yellow'
    else:
        text_color, message, color = (0,255,0), "Detected Signal Status: Go", 'green'
    cv2.putText(image, message,(15,70),font, font_scale+0.5,text_color,font_thickness+1,cv2.LINE_AA)
    cv2.putText(image, 34*'-',font,font_scale,(255,255,255),font_thickness,cv2.LINE_AA)
    return image,color

class LineDetector:
    def __init__(self,num_frames_avg=10):
        self.y_start_queue = deque(maxlen=num_frames_avg)
        self.y_end_queue = deque(maxlen=num_frames_avg)
    
    def detect_white_line(self,frame,color,slope1 = 0.03,intercept1 = 920, slope2 =0.03,intercept2 =770, slope3 = -0.8,intercept3 = 2420):
        def get_color_code(c):
             x = {'red':(0,0,255),'green':(0,255,0),'yellow':(0,255,255)}
             return x.get(c.lower())
        frame_org = frame.copy()
        def line1(x):
            return slope1*x + intercept1
        def line2(x):
            return slope2*x + intercept2
        def line3(x):
            return slope3*x + intercept3
        h,w,_ = frame.shape
        
        





