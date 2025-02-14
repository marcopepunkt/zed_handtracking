import cv2
import numpy as np
import pyzed.sl as sl
import sys
import os
import yaml
import open3d as o3d
import time
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from visualizer import open3d_visualizer
from zed_util import MultiCamSync, CameraData

origin_id = 39725782 # The serial number of the camera at the origin of the world frame
NUM_BLOBS = 3
color_dict = {39725782: (255, 0, 0), 38580376: (0, 255, 0), 0: (0, 0, 255)}
blob_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]



class Blobs:
    def __init__(self, camera_id, num_blobs):
        self.camera_id = camera_id
        self.uv = np.zeros(shape=(num_blobs,2)) # u,v coordinates in the image frame
        self.xyz_stereo_estimate = np.zeros(shape=(num_blobs,3)) # in the global frame
        self.xyz_triangulated = None # in the global frame
        self.num_blobs = num_blobs # fixed number of blobs 
    
    def filter_blobs(self, new_uv, new_xyz):
        if len(new_uv) < self.num_blobs or len(new_uv) != len(self.uv):
            print(f"Expected {self.num_blobs} blobs, but only found {len(new_uv)}. Need to rematch markers")
            global rematch_markers
            rematch_markers = True
            
            
        # Assign the blobs to the correct blobid based on the distance to the previous blob
        # the goal is that the total distance between the blobs is minimized
        # Build cost matrix (N x M)
        cost = cdist(self.uv, new_uv) # shape: [N, M]
        
        # Solve the assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Update matched blobs
        for i, j in zip(row_ind, col_ind):
            self.uv[i] = new_uv[j]
            self.xyz_stereo_estimate[i] = new_xyz[j]
        
        # solve
        return self.uv, self.xyz_stereo_estimate
        
        
        

def blob_detection(image_np, cam_id):
    """This function detects blobs in an image and returns their centers"""
    # Convert to HSV for easier color thresholding
    hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

    #cv2.imshow(f"HSV {cam_id}", hsv)
    
    # Define orange color range (tune these as needed)
    lower_orange = np.array([0, 224, 150])
    upper_orange = np.array([255, 255, 227])

    # Threshold to get only orange
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    #cv2.imshow(f"Theshhold Mask {cam_id}", mask)

    # Clean up noise via opening/closing
    kernel = np.ones((3, 3), np.uint8)
    
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    #cv2.imshow(f"Cleaned Mask {cam_id}", mask)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centers = []
    for cnt in contours:
        # Approx. the sphere with a minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius > 5:  # filter out too-small circles
            centers.append((int(x), int(y)))

    return centers
 


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
    depth_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.F32_C1)    
  
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
            
            center = blob_detection(image_np)
            
            if center is not None:
                cv2.circle(image_np, center, 5, (0, 0, 255), -1)
                # Get the depth value at the center of the blob
                depth_value = float(depth_np[center[1], center[0]])
                cv2.putText(image_np, f"Distance: {depth_value} mm", (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                print(f"Blob center at {center} has depth: {depth_value} mm")

            # Show the color in the center of the image
            center = (image_size.width // 2, image_size.height // 2)
            print(f"Center of the image has color: {image_np[center[1], center[0]]}")
            cv2.circle(image_np, center, 5, (0, 0, 255), -1)
                       
            cv2.imshow("Blob Detection", image_np)
            
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to grab frame")
    # Close the camera
    zed.close()
    cv2.destroyAllWindows()


def triangulation(cams, uv):
    """This function returns the 3D position of a point in the global frame (from calibration pattern) from a single image"""
    assert len(cams) == len(uv), "The number of cameras and the number of uv coordinates must be the same"
    assert len(cams) ==2 , "The number is so far 2, TODO for more than 2 cameras" # TODO: Implement for more than 2 cameras
    assert len(uv[0]) == len(uv[1]), "The uv coordinates must be a 2D array"
    
    cam1, cam2 = cams[0], cams[1]
    P1 = cam1.get_projection_matrix()
    P2 = cam2.get_projection_matrix()

    # Convert list of (u,v) to arrays of shape (2, N)
    pts1 = np.array(uv[0]).T  # shape (2, N)
    pts2 = np.array(uv[1]).T  # shape (2, N)
    
    # divide uv coodinates by with and height to get normalized coordinates -> not needed, since incorparated in the projection matrix
    # pts1[0] = pts1[0] / cam1.width
    # pts1[1] = pts1[1] / cam1.height
    # pts2[0] = pts2[0] / cam2.width
    # pts2[1] = pts2[1] / cam2.height
    
    

    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
    points_3d = (points_4d[:3] / points_4d[3]).T  # shape (N, 3)
    
    return points_3d
    
    
def load_camera_calib(id = None, path = "calibration_output"):
    """This function loads the camera calibration parameters from a yaml file. If an ID 
    is provided, it will only load the calibration parameters for that camera"""
    #iterate over all files in the folder calibration_output
    cams = []
    for idx,file in enumerate(os.listdir(path)):
        #open the file
        fullpath = os.path.join(path, file)
        with open(fullpath, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                if id is not None and data["id"] != id:
                    continue
                #create a camera object
                cams.append(CameraData(data["id"],data['intr_3x3'],data['extr_4x4']))
            except yaml.YAMLError as exc:
                print(exc)
    return cams
    
def assign_markers(blobs_dict, cams):
    """
    1) Gathers 3D points (already transformed into a common/world frame).
    2) Matches them using Hungarian in 3D space.
    3) If distance > distance_threshold, skip the match and fallback to the camera's stereo data.
    4) Returns final matched pairs and fallback points.

    Assumes:
    - blobs_dict[camX_id].xyz_stereo_estimate has shape [N, 3] in same coordinate frame.
    """
    global rematch_markers
    rematch_markers = False
    # 1) Gather camera 3D points
    xyz_cam = []
    for cam in cams: # For each camera
        cam_id = cam.camera_id
        xyz_stereo_point = np.array(blobs_dict[cam_id].xyz_stereo_estimate)
        # Replace nan with 0 (if any)
        if np.isnan(xyz_stereo_point).any():
            print(f"Found NaN in the stereo estimate for camera {cam_id}")
            rematch_markers = True
        xyz_stereo_point = np.nan_to_num(xyz_stereo_point)
        xyz_cam.append(xyz_stereo_point)
   
   
    # 2) Build cost matrix from 3D distances
    cost_matrix = cdist(xyz_cam[0], xyz_cam[1])  # shape: [N1, N2]

    # 3) Hungarian assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    marker_id_dict = {
        cams[0].camera_id: list(row_ind),
        cams[1].camera_id: list(col_ind)
    }
    
    
    
    return marker_id_dict


if __name__ == "__main__":
    # Load the calibration data
    global rematch_markers
    rematch_markers = True
    marker_id_dict = {}
    blobs_dict = {}
    cams = load_camera_calib()
    for cam in cams: 
        cam.init_zed(f"cpp_multicam_rec/build/SVO_SN{cam.camera_id}.svo2")
        blobs_dict[cam.camera_id] = Blobs(cam.camera_id, NUM_BLOBS)
        marker_id_dict[cam.camera_id] = [i for i in range(NUM_BLOBS)]
    
    frame_grabber = MultiCamSync(cams)
    
    vis = open3d_visualizer(cams)
    print("Initialized cameras")    

    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    xyz_mat = sl.Mat()
    loop = True
    while loop:
        img_list = []
        if frame_grabber.grab_frames() != sl.ERROR_CODE.SUCCESS:
            print("Failed to grab frames")
            loop = False
            break
        
        matched_blobs_uv = []
        
        for cam in cams:
            cam.zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            cam.zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            cam.zed.retrieve_measure(xyz_mat, sl.MEASURE.XYZ)
            #print(f"Cam {cam.camera_id} at timestamp {cam.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()}")
            
            id = cam.camera_id
            
            img = image_mat.get_data()[:, :, :3] 
            img = img.astype(np.uint8)   
            img_list.append(img)
            depth = depth_mat.get_data()
            xyz_data = xyz_mat.get_data()         
            transform = cam.extrinsics
            
            # save image as png to select the hsv values
            cv2.imwrite(f"image_{id}.png", img)
            
            
          
            blob_centers = blob_detection(img, id)
            if len(blob_centers) != 0:
                # Do the blob detection here
                centers_as_int = []
                xyzs_global = []
                for center in blob_centers:
                    center_as_int = np.array(center, dtype=int)
                    centers_as_int.append(center_as_int)
                    depthval = depth[center_as_int[1], center_as_int[0]]
                    
                    # Move the xyz point 10 mm further back
                    xyz_stereo_point = xyz_data[center_as_int[1], center_as_int[0]]
                    xyz_stereo_point = np.array([xyz_stereo_point[0], xyz_stereo_point[1], xyz_stereo_point[2], 1])
                    direction = xyz_stereo_point[0:3] / np.linalg.norm(xyz_stereo_point[0:3])
                    xyz_stereo_point[0:3] = xyz_stereo_point[0:3] + 10 * direction     
                    xyz_global = np.dot(np.linalg.inv(transform), xyz_stereo_point)
                    xyzs_global.append(xyz_global[:3])
             
                matched_uvs, matched_xyzs = blobs_dict[cam.camera_id].filter_blobs(centers_as_int, xyzs_global)
                vis.visualize_points(xyzs_global, color = color_dict[id])

                # Sort the matched uvs and xyzs based on the marker id
                matched_uvs = matched_uvs[marker_id_dict[cam.camera_id]]
                matched_xyzs = matched_xyzs[marker_id_dict[cam.camera_id]]
                
                matched_blobs_uv.append(matched_uvs)

                for blob_id, matched_uv in enumerate(matched_uvs):
                    cv2.circle(img, (int(matched_uv[0]), int(matched_uv[1])), 5, blob_colors[blob_id], -1)
                    #cv2.putText(img, f"At point {matched_xyzs[blob_id]}", (int(matched_uv[0]) + 10 , int(matched_uv[1]) -10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                #uv.append(centers)
            cv2.imshow(f"Camera {id}", img)

        # Now we have the uv coordinates and the xyz stereo coordinates from each camera
        # We now match the points and triangulate the assigned ones. We use the xyz stereo coordinates, if we cannot assign them 
        # as an input we have the list of blob classes and the output is the list of xyz global coordinates and the list of uv coordinates
        # for each camera
        
        if rematch_markers:
            marker_id_dict = assign_markers(blobs_dict, cams)
        
        if len(matched_blobs_uv) ==2: 
            xyz_points = triangulation(cams, matched_blobs_uv)
            vis.visualize_points(xyz_points, color = (0, 0, 0))
        else: 
            print("Not enough cameras to triangulate")     

        
        cv2.waitKey(1)  # Allow OpenCV to update the window
        #time.sleep(1)
    # Then get the uv coordinates from the image for the blob and call 
    print("done")
    #main()
    