import cv2
import numpy as np
import pyzed.sl as sl
import sys


def main():
    # Create a ZED camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD1080
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    # Open the camera
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera")
        exit(1)

    image_size = zed.get_camera_information().camera_configuration.resolution
    image_size.width = image_size.width 
    image_size.height = image_size.height

    # Create Mat objects to hold images and depth data
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    # Create a blob detector with default parameters
    color_lower = (0, 130, 130,0)
    color_upper = (90, 200, 200,255)   
    
    detector = cv2.SimpleBlobDetector.create()

    while True:
        # Grab an image and depth data
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            #print("Grabbed frame")
            # Retrieve left image
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            # Retrieve depth map
            zed.retrieve_image(depth_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)

            #print("Retrieved image and depth")
            # Convert sl.Mat to numpy array
            image_np = image_zed.get_data()
            depth_np = depth_zed.get_data()

            #cv2.imshow("Image", image_np)
            #cv2.imshow("Depth", depth_np)
            
            # select relevant colors in the image
            #processed_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            mask = cv2.inRange(image_np, color_lower, color_upper)
            
            cv2.imshow("Mask", mask)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                                    cv2.CHAIN_APPROX_NONE)
            # Draw contours on the image
            #cv2.drawContours(image_np, contours, -1, (0, 255, 0), 2)
            
            # Get the RGB value of the pixel in the middle of the image
            # middle_x, middle_y = image_size.width // 2, image_size.height // 2
            # rgb_value = image_np[middle_y, middle_x]
            # print(f"RGB value at the center ({middle_x}, {middle_y}): {rgb_value}")
            
            # # Draw a circle at the center of the image
            # cv2.circle(image_np, (middle_x, middle_y), 10, (0, 0, 255), 2)
            
            blob = max(contours, key=lambda el: cv2.contourArea(el))
            print(cv2.contourArea(blob))
            if cv2.contourArea(blob) > 1000:    
                M = cv2.moments(blob)
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                cv2.circle(image_np, center, 10, (0, 0, 255), -1)
                # Get the depth value at the center of the blob
                depth_value = depth_np[center[1], center[0]]
                cv2.putText(image_np, f"Distance: {depth_value} mm", (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                print(f"Blob center at {center} has depth: {depth_value} mm")


            #processed_image = cv2.bitwise_and(image_np, image_np, mask=mask)
            #cv2.imshow("Processed Image", processed_image)
            # Detect blobs
            #keypoints = detector.detect(processed_image)
            #print(keypoints)

            # Draw detected blobs as red circles
            #im_with_keypoints = cv2.drawKeypoints(image_np, keypoints,  None, (255, 0, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            # Display the image with keypoints
            cv2.imshow("Blob Detection", image_np)

            # Get depth value for each keypoint
            """for keypoint in keypoints:
                x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
                depth_value = depth_np[y, x]
                print(f"Blob at ({x}, {y}) has depth: {depth_value} mm")"""

            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to grab frame")
    # Close the camera
    zed.close()
    cv2.destroyAllWindows()

def hello_zed():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.AUTO # Use HD720 opr HD1200 video mode, depending on camera type.
    init_params.camera_fps = 30 # Set fps at 30
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
        
   # Capture 50 frames and stop
    i = 0
    image = sl.Mat()
    while (i < 50) :
        # Grab an image
        if (zed.grab() == sl.ERROR_CODE.SUCCESS) :
            # A new image is available if grab() returns SUCCESS
            zed.retrieve_image(image, sl.VIEW.LEFT) # Get the left image
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE) # Get the timestamp at the time the image was captured
            print("Image resolution: {0} x {1} || Image timestamp: {2}\n".format(image.get_width(), image.get_height(),
                    timestamp.get_milliseconds()))
            i = i+1
    # Get camera information (ZED serial number)
    #zed_serial = zed.get_camera_information().serial_number
    #print("Hello! This is my serial number: {0}".format(zed_serial))

    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()