import cv2
import numpy as np
import pyzed.sl as sl
import sys
import os
import yaml

origin_id = 39725782 # The serial number of the camera at the origin of the world frame

class camera_data:
    def __init__(self, camera_id, intrinsics, extrinsics):
        self.camera_id = camera_id
        self.intrinsics = np.array(intrinsics)
        self.extrinsics = np.array(extrinsics)
        
    def get_projection_matrix(self):
        return self.intrinsics @ self.extrinsics[:3, :]

def main():
    # Create a ZED camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720 
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
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
    color_lower = (0, 130, 130,0) # BGR format (This is for yellow color)
    color_upper = (90, 200, 200,255)   
    
    detector = cv2.SimpleBlobDetector.create()

    while True:
        # Grab an image and depth data
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            #print("Grabbed frame")
            # Retrieve left image
            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            # Retrieve depth map
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH, sl.MEM.CPU, image_size)

            #print("Retrieved image and depth")
            # Convert sl.Mat to numpy array
            image_np = image_zed.get_data()
            depth_np = depth_zed.get_data()

            #cv2.imshow("Image", image_np)
            cv2.imshow("Depth", depth_np)
            
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
            if cv2.contourArea(blob) > 500:    
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


def triangulation(cams, uv):
    """This function returns the 3D position of a point in the camera frame from a single image"""
    assert len(cams) == len(uv), "The number of cameras and the number of uv coordinates must be the same"
    assert len(cams) ==2 , "The number is so far 2, TODO for more than 2 cameras" # TODO: Implement for more than 2 cameras
    
    points_3d = []
    
    i, j = 0, 1
    
    
    # Set the camera with the origin_id as cam 1
    if cams[0].camera_id != origin_id:
        i, j = 1, 0
        print("Swapped the cameras, since the first camera is not the origin camera")
            
    
    # Get the cameras and the uv coordinates
    cam1, cam2 = cams[i], cams[j]
    x1, x2 = np.array(uv[i]), np.array(uv[j])
    
        
    # Construct projection matrices P1 and P2
    P1 = cam1.get_projection_matrix()
    P2 = cam2.get_projection_matrix()
    
   
    # Triangulate 3D point using OpenCV
    point_4d = cv2.triangulatePoints(P1, P2, x1, x2)
    
    # Convert from homogeneous to Cartesian coordinates
    p_W = point_4d[:3] / point_4d[3]
    
    # Transform to the first camera's coordinate frame
    R_W_C1 = np.linalg.inv(cam1.extrinsics[:3, :3])
    p_target = R_W_C1 @ (p_W.flatten() - cam1.extrinsics[:3, 3])
    
    points_3d.append(p_target)
    
    return points_3d
    
def load_camera_calib(path = "calibration_output") -> list[camera_data]:
    """This function loads the camera calibration parameters from a yaml file"""
    #iterate over all files in the folder calibration_output
    cams = []
    for idx,file in enumerate(os.listdir(path)):
        #open the file
        fullpath = os.path.join(path, file)
        with open(fullpath, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                #create a camera object
                cams.append(camera_data(data["id"],data['intr_3x3'],data['extr_4x4']))
            except yaml.YAMLError as exc:
                print(exc)
    return cams
        
if __name__ == "__main__":
    cams = load_camera_calib()
    
    # Get the uv coordinates -> Must be normalized
    uv1 = [612/1280, 395/720]
    uv2 = [736/1280, 331/720]
    
    point = triangulation(cams, [uv1, uv2])
    print(point)
    # Then get the uv coordinates from the image for the blob and call 
    print("done")
    #main()
    