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

origin_id = 39725782 # The serial number of the camera at the origin of the world frame
NUM_BLOBS = 3
color_list = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
rematch_markers = True # this is a flag to indicate if the markers should be rematched

class CameraData:
    def __init__(self, camera_id, intrinsics, extrinsics):
        self.camera_id = camera_id
        self.intrinsics = np.array(intrinsics)
        self.extrinsics = np.array(extrinsics)
        
    def get_projection_matrix(self):
        return self.intrinsics @ self.extrinsics[:3, :]
   
    def init_zed(self, svo_input_path):
            # loads the svo2 file and returns the frames of the depth and the left color image
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(svo_input_path)
        init_params.svo_real_time_mode = False  # Don't convert in realtime
        init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL 
        init_params.camera_resolution = sl.RESOLUTION.HD1080
        
        zed = sl.Camera()

        # Open the SVO file specified as a parameter
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            sys.stdout.write(repr(err))
            zed.close()
            exit()
            
        self.zed = zed
    

class Blobs:
    def __init__(self, camera_id, num_blobs):
        self.camera_id = camera_id
        self.uv = np.zeros(shape=(num_blobs,2)) # u,v coordinates in the image frame
        self.xyz_stereo_estimate = np.zeros(shape=(num_blobs,3)) # in the global frame
        self.xyz_triangulated = None # in the global frame
        self.num_blobs = num_blobs # fixed number of blobs 
    
    def filter_blobs(self, new_uv, new_xyz):
        if len(new_uv) < self.num_blobs:
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
        
class Markers: 
    def __init_(self, num_markers):
        """Each blob is tracked via the Blobs class, but to match the Blobs across 
        mutiple cameras, we need to track the markers in 3D and return a indexlist, 
        which assigns each blob to a marker. This is necessary for triangulation."""
        self.num_markers = num_markers
        self.xyz = np.zeros(shape=(num_markers,3))
    
    def match_markers(self, uv1, uv2, xyz1, xyz2):
        """This function matches the points"""
        cost = cdist(self.xyz, xyz_new)
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # for i, j in zip(row_ind, col_ind):
        #     self.xyz[i] = xyz_new[j]
            
        return zip(row_ind, col_ind)
    
    def reassign_markers(self, camera_id):
        """This function sorts the blobs so that they match between different cameras"""
        
        
        return assignment 
        
        
        

def blob_detection(image_np, cam_id):
    """This function detects blobs in an image and returns their centers"""
    # Convert to HSV for easier color thresholding
    hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

    #cv2.imshow(f"HSV {cam_id}", hsv)
    
    # Define orange color range (tune these as needed)
    lower_orange = np.array([0, 224, 150])
    upper_orange = np.array([255, 255, 196])

    # Threshold to get only orange
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    
    cv2.imshow(f"Theshhold Mask {cam_id}", mask)

    # Clean up noise via opening/closing
    kernel = np.ones((5, 5), np.uint8)
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
    
    points_3d = []
    
    i, j = 0, 1
    
    # Get the cameras and the uv coordinates
    cam1, cam2 = cams[i], cams[j]
        
    K1, K2 = cam1.intrinsics, cam2.intrinsics

    # Convert pixel coordinates (u, v) to normalized image coordinates (x_n, y_n)
    x1 =  np.array([uv[i][0], uv[i][1], 1])
    x2 = np.array([uv[j][0], uv[j][1], 1])
        
    # Construct projection matrices P1 and P2
    P1 = cam1.get_projection_matrix()
    P2 = cam2.get_projection_matrix()
    
    # Triangulate 3D point using OpenCV
    point_4d = cv2.triangulatePoints(P1, P2, x1[:2], x2[:2])

    # Convert from homogeneous to Cartesian coordinates
    p_W = point_4d[:3] / point_4d[3]
    p_W = p_W.reshape(-1)
    return p_W
    
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
    

if __name__ == "__main__":
    # Load the calibration data
    blobs_dict = {}
    cams = load_camera_calib()
    for cam in cams: 
        cam.init_zed(f"cpp_multicam_rec/build/SVO_SN{cam.camera_id}.svo2")
        blobs_dict[cam.camera_id] = Blobs(cam.camera_id, NUM_BLOBS)
        
        
    print("Initialized cameras")    

    marker = Markers(NUM_BLOBS)
    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    xyz_mat = sl.Mat()
    loop = True
    while loop:
        for cam in cams:
            if cam.zed.grab() == sl.ERROR_CODE.SUCCESS:
                cam.zed.retrieve_image(image_mat, sl.VIEW.LEFT)
                cam.zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
                cam.zed.retrieve_measure(xyz_mat, sl.MEASURE.XYZ)
            else:
                print("Failed to grab frame")
                loop = False
                break
            id = cam.camera_id
            
            img = image_mat.get_data()[:, :, :3] 
            img = img.astype(np.uint8)   
            depth = depth_mat.get_data()
            xyz_data = xyz_mat.get_data()         
            transform = cam.extrinsics
            
            # save image as png to select the hsv values
            #cv2.imwrite(f"image_{id}.png", img)
            
            
          
            blob_centers = blob_detection(img, id)
            if blob_centers is not None:
                # Do the blob detection here
                centers_as_int = []
                xyzs_global = []
                for center in blob_centers:
                    center_as_int = np.array(center, dtype=int)
                    centers_as_int.append(center_as_int)
                    depthval = depth[center_as_int[1], center_as_int[0]]
                    
                    # Move the xyz point 10 mm further back
                    xyz_stereo_point = xyz_data[center_as_int[1], center_as_int[0]]
                    direction = xyz_stereo_point[0:3] / np.linalg.norm(xyz_stereo_point[0:3])
                    xyz_stereo_point[0:3] = xyz_stereo_point[0:3] + 10 * direction     
                    xyz_global = np.dot(np.linalg.inv(transform), xyz_stereo_point)
                    xyzs_global.append(xyz_global[:3])
                                   
                
                matched_uvs, matched_xyzs = blobs_dict[cam.camera_id].filter_blobs(centers_as_int, xyzs_global)
                
                assignment = marker.reassign_markers(cam.camera_id)
                assigned_uvs = [matched_uvs[i] for i in assignment]
                assigned_xyzs = [matched_xyzs[i]for i in assignment]
                
                
                
                for blob_id, matched_uv in enumerate(matched_uvs):
                    cv2.circle(img, (int(matched_uv[0]), int(matched_uv[1])), 5, color_list[blob_id], -1)
                    cv2.putText(img, f"At point {matched_xyzs[blob_id]}", (int(matched_uv[0]) + 10 , int(matched_uv[1]) -10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                #uv.append(centers)
                            
            cv2.imshow(f"Camera {id}", img)
            #cv2.imshow(f"Depth {id}", depth)
        
        # Now we have the uv coordinates for each blob and triangulate them using opencv
        for cam in cams:
            uv = blobs_dict[cam.camera_id].uv
            xyz_global = triangulation(cams, uv)
            marker.xyz[cam.camera_id] = xyz_global
        
        
        if rematch_markers: 
            print("Rematching markers")
            marker.match_markers(blobs_dict)
            rematch_markers = False
        
        cv2.waitKey(1)  # Allow OpenCV to update the window
        #time.sleep(1)
    # Then get the uv coordinates from the image for the blob and call 
    print("done")
    #main()
    