import cv2
import numpy as np;

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
cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0,255, nothing)
cv2.createTrackbar("LS", "Tracking", 225, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 122, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 15, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)
cv2.createTrackbar("Brightness", "Tracking", 100, 200, nothing)  # 100 = original brightness

# Read test image
frame = cv2.imread("/home/aidara/augmented_imitation_learning/data_storage/threecolor_movement/raw_images/39725782_0.png")

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
    
    bounding_box_robot_base = ((987,331), (1176,482)) # Top left and bottom right corners
    mask = cv2.rectangle(mask, bounding_box_robot_base[0], bounding_box_robot_base[1], (0, 0, 0), -1)
    
    
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