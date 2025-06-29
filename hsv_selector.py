import cv2
import numpy as np
from blob_detector import get_cam_config
from zed_util import load_camera_calib
import argparse

# Just dummy function for callbacks from trackbar
def nothing(x):
    pass

# of the points clicked on the image
def click_event(event, x, y, flags, params):
   if event == cv2.EVENT_LBUTTONDOWN:
      print(f'({x},{y})')
      
    #   # put coordinates as text on the image
    #   cv2.putText(img, f'({x},{y})',(x,y),
    #   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
      
    #   # draw point on the image
    #   cv2.circle(img, (x,y), 3, (0,255,255), -1)

# Create a trackbar window to adjust the HSV values
# They are preconfigured for a yellow object 
cam = 32689769
calib_path = "/home/aidara/augmented_imitation_learning/zed_handtracking/calibration_output/20250629_112335"
args = argparse.Namespace(calib_path=calib_path)
cams = load_camera_calib(calib_path)
hsv_limits = get_cam_config(args, cams)
frame = cv2.imread(f"/home/aidara/augmented_imitation_learning/data_storage/pos_test/test_movement_3_Episode_0/raw_images/{cam}_raw.png")
finger ="index_tip" # "thumb" ,"index_base", "index_tip"
hsv_limits = hsv_limits[finger][cam]
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", hsv_limits[0][0],255, nothing)
cv2.createTrackbar("LS", "Tracking", hsv_limits[0][1],255, nothing)
cv2.createTrackbar("LV", "Tracking", hsv_limits[0][2],255, nothing)
cv2.createTrackbar("UH", "Tracking", hsv_limits[1][0],255, nothing)
cv2.createTrackbar("US", "Tracking", hsv_limits[1][1],255, nothing)
cv2.createTrackbar("UV", "Tracking", hsv_limits[1][2],255, nothing)
cv2.createTrackbar("Brightness", "Tracking", 100, 200, nothing)  # 100 = original brightness

# Read test image

cv2.setMouseCallback('Tracking', click_event)

while True:
    # Convert to HSV colour space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Read the trackbar values
    lh = cv2.getTrackbarPos("LH", "Tracking")
    ls = cv2.getTrackbarPos("LS", "Tracking")
    lv = cv2.getTrackbarPos("LV", "Tracking")
    uh = cv2.getTrackbarPos("UH", "Tracking")
    us = cv2.getTrackbarPos("US", "Tracking")
    uv = cv2.getTrackbarPos("UV", "Tracking")
    brightness = cv2.getTrackbarPos("Brightness", "Tracking")
    
    alpha = brightness / 100  # scale from 0.0 to 2.0
    bright_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)


    # Create arrays to hold the minimum and maximum HSV values
    hsvMin = np.array([lh, ls, lv])
    hsvMax = np.array([uh, us, uv])
    
    # Apply HSV thresholds 
    mask = cv2.inRange(hsv, hsvMin, hsvMax)
    
    
    
    # Uncomment the lines below to see the effect of erode and dilate
    #mask = cv2.erode(mask, None, iterations=5)
    #mask = cv2.dilate(mask, None, iterations=5)

    # The output of the inRange() function is black and white
    # so we use it as a mask which we AND with the orignal image
    res = cv2.bitwise_and(bright_frame, bright_frame, mask=mask)

    # Show the result
    cv2.imshow("Result view", res)

    # Wait for the escape key to be pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()