from typing import Dict
import json
import argparse
import os

import cv2
import numpy as np
import pyzed.sl as sl

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from visualizer import open3d_visualizer, CoordFrameVis
from zed_util import MultiCamSync, CameraData, load_camera_calib
from robot_simulator import RobotSim

NUM_BLOBS = 3
color_dict = {39725782: (221, 0, 252), 38580376: (0, 255, 238)}
blob_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

class Blobs:
    def __init__(self, camera_id, num_blobs):
        self.camera_id = camera_id
        self.uv = [] # u,v coordinates in the image frame
        self.xyz_stereo_estimate = [] # in the global frame
        self.xyz_triangulated = [] # in the global frame
        self.num_blobs = num_blobs # fixed number of blobs 
    
    def len(self):
        return len(self.uv)
    
    def filter_blobs(self, new_uv, new_xyz):
        """This function tracks the individual blobs in the 2D image frame"""
        if len(new_uv) != len(self.uv):
            print(f"Expected {self.num_blobs} blobs, but only found {len(new_uv)}. Need to rematch markers")
            global rematch_markers
            rematch_markers = True
            
        if len(self.uv) == 0:
            self.uv = new_uv 
            self.xyz_stereo_estimate = new_xyz
            print("There are no previous blobs, nothing to track")
        else: 
            cost = cdist(self.uv, new_uv)
            row_ind, col_ind = linear_sum_assignment(cost)

            matched_old = set(row_ind)
            matched_new = set(col_ind)

            # Rebuild arrays only with matched pairs
            new_uv_list = []
            new_xyz_list = []

            for i, j in zip(row_ind, col_ind):
                new_uv_list.append(new_uv[j])
                new_xyz_list.append(new_xyz[j])

            # Add any leftover new points
            unmatched_new = set(range(len(new_uv))) - matched_new
            for idx in unmatched_new:
                new_uv_list.append(new_uv[idx])
                new_xyz_list.append(new_xyz[idx])

            # Update self.uv / self.xyz_stereo_estimate with the new combined set
            self.uv = np.array(new_uv_list)
            self.xyz_stereo_estimate = np.array(new_xyz_list)

            assert len(self.uv) == len(new_uv), "Just some sanity check of the algorithm"
            # do a check if all the elements of new_uv are in self.uv
            assert all([x in self.uv for x in new_uv]), "All the elements of new_uv should be in self.uv"
                        
        # This is returned for visualization purposes
        return self.uv
   
    
class Markers: 
    def __init__(self, blob_dict, cams):
        self.cams = cams
        for cam_id, blobs in blob_dict.items():
            if blobs.len() != 3:
                print("For successful initialization all 3 Blobs must be seen from both cameras") 
                raise ValueError("Not all blobs are seen") 
        
        self.unmatched_dict = {} # Since all must be seen, this is empty
        # Now we have blob pairs that make up markers. 
        pairs = self.match_blobs_to_markers(blob_dict)
        
        
        # We can now do triangulation to get the 3D position of the markers
        unsorted_keypoints = self.triangulation(pairs)
        
        self.keypoints = unsorted_keypoints
        
        self.offset_dict= {}
        for cam in self.cams:
            offset = np.mean(blob_dict[cam.camera_id].xyz_stereo_estimate[self.marker_id_dict[cam.camera_id]]-self.keypoints, axis = 0)
            if np.isnan(offset).any():
                print(f"Found NaN in the offset for camera {cam.camera_id}")
                raise ValueError("Found NaN in the offset") 
            self.offset_dict[cam.camera_id] = offset

        assert len(self.keypoints) == 3, "The number of keypoints must be 3"
        print("Initialized markers sucessfully")

        
    def track_markers(self, blob_dict): 
        """This function tracks the markers"""
        # If the marker number has not changed, we dont have to match the markers again
        global rematch_markers
        
        unmatched_dict = {}
        
        if rematch_markers:
            matched_pairs = self.match_blobs_to_markers(blob_dict)
        else: 
            matched_pairs = []
            
            for cam in self.cams: # For each camera
                # Load UV coordinates of the blobs
                uv = blob_dict[cam.camera_id].uv
                # Assign the blobs to one another to form th markers. 
                uv_indices = self.marker_id_dict[cam.camera_id]            
                matched_pairs.append(uv[uv_indices])
            
        
            
        unsorted_keypoints = self.triangulation(matched_pairs)
        
        
        # This is the calibrated stereo estimate of the markers
        for cam_id, unmatched_indices in self.unmatched_dict.items():
            for idx in unmatched_indices:
                estimation = blob_dict[cam_id].xyz_stereo_estimate[idx] + self.offset_dict[cam_id]
                if np.isnan(estimation).any():
                    print(f"Cannot estimate the depth of camera {cam_id}")
                    
                    # TODO: Just take the average of the other points and add the offset
                else:
                    unsorted_keypoints = np.vstack((unsorted_keypoints, estimation))
        
        #assert len(unsorted_keypoints) == 3, "The number of keypoints must be 3"
        
        cost_matrix = cdist(self.keypoints, unsorted_keypoints)
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Update the keypoints
        for i, j in zip(row_ind, col_ind):
            self.keypoints[i] = unsorted_keypoints[j]
        
        # assign the markers with the closest distance to the previous markers, just set it to 0 
        return self.keypoints
    
    def match_blobs_to_markers(self, blobs_dict):
        """
        This function matches the blobs from all cameras and creates a dictionary that
        matches the blobs to one another e.g. 
        Blob 1 from camera 1 is matched to blob 2 from camera 2
        blob 2 from camera 1 is matched to blob 1 from camera 2
        blob 3 from camera 1 is matched to blob 3 from camera 2
        -> {2139430: [0, 1, 2], 1234235: [1, 0, 2]}
        """
        global rematch_markers
        rematch_markers = False
        # 1) Gather camera 3D points
        xyz_cam = []
        uv_cam = []
        for cam in self.cams: # For each camera
            cam_id = cam.camera_id
            xyz_stereo_point = np.array(blobs_dict[cam_id].xyz_stereo_estimate)
            # Replace nan with 0 (if any)
            if np.isnan(xyz_stereo_point).any():
                print(f"Found NaN in the stereo estimate for camera {cam_id}")
                rematch_markers = True
            xyz_stereo_point = np.nan_to_num(xyz_stereo_point)
            xyz_cam.append(xyz_stereo_point)
            uv_cam.append(blobs_dict[cam_id].uv)
    
        #assert len(xyz_cam[0]) == len(xyz_cam[1]), "The number of blobs must be the same for both cameras"
        
        # 2) Build cost matrix from 3D distances
        cost_matrix = cdist(xyz_cam[0], xyz_cam[1])  # shape: [N1, N2]

        # 3) Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assert len(row_ind) == len(col_ind), "The number of rows and columns must be the same"
        # This is really ticky, they have to be the same length and not all xyzs are used. EG. we have 3 blobs in camera 1 and 2 blobs in camera 2
        # Then we just have some row_ind and col_ind like that [1,2] and [0,1] 
        
        self.marker_id_dict = {
            self.cams[0].camera_id: list(row_ind),
            self.cams[1].camera_id: list(col_ind)
        }
        # Now i want to return the pairs of blobs that make up the markers
        pairs = []
        for cam in self.cams:
            uv = blobs_dict[cam.camera_id].uv
            uv_indices = self.marker_id_dict[cam.camera_id]
            pairs.append(uv[uv_indices])
            
        unmatched_dict = {}
        unmatched_indices = set(range(len(uv_cam[1]))) - set(self.marker_id_dict[self.cams[1].camera_id])
        if len(unmatched_indices) != 0:
            unmatched_dict[self.cams[1].camera_id] = unmatched_indices
            
        unmatched_indices = set(range(len(uv_cam[0])))- set(self.marker_id_dict[self.cams[0].camera_id])
        if len(unmatched_indices) != 0:
            unmatched_dict[self.cams[0].camera_id] = unmatched_indices
        
        self.unmatched_dict = unmatched_dict    
        
        return pairs

    def triangulation(self, uv):
        """This function returns the 3D position of a point in the global frame (from calibration pattern) from a single image"""
        assert len(self.cams) ==2 , "The number is so far 2, TODO for more than 2 cameras" # TODO: Implement for more than 2 cameras
        assert len(uv[0]) == len(uv[1]), "The uv coordinates have the same length"
        #print(f"--- ES EXISTIEREN {len(uv[0])}")
        
        cam1, cam2 = self.cams[0], self.cams[1]
        P1 = cam1.get_projection_matrix()
        P2 = cam2.get_projection_matrix()

        # Convert list of (u,v) to arrays of shape (2, N)
        pts1 = np.array(uv[0], dtype=float).T  # shape (2, N)
        pts2 = np.array(uv[1], dtype=float).T  # shape (2, N)
        
        X = cv2.triangulatePoints( P1, P2, pts1, pts2)
        # Remember to divide out the 4th row. Make it homogeneous
        X /= X[3]
        #print(X)
        assert X.shape[1] == len(uv[0]), "The number of points must be the same"
        
        return X[:3].T
          
    def get_hand_pose(self):
        base = np.mean(self.keypoints[1:3], axis = 0)
        finger_base = self.keypoints[1]
        finger_tip = self.keypoints[0]
        thumb = self.keypoints[2]
        foward_vector = self._normal_to_line(thumb, finger_base, finger_tip)
        forward_point = base + foward_vector
        return self._create_homogeneous_matrix(base, foward_vector + base, thumb), forward_point
    
    def _normal_to_line(self, P1, P2, P3):
        """
        Calculate the normal vector of a line defined by two points and a third point.       
        Parameters:
            P1, P2, P3 : tuple or list of three coordinates (x, y, z)
            P1 and P2 define the line, P3 is the point
        
        Returns:
            normal_vector : numpy array representing the normal vector
            unit_normal_vector : numpy array representing the unit normal vector
        """
        # Convert to numpy arrays
        P1, P2, P3 = np.array(P1), np.array(P2), np.array(P3)
        
        # Direction vector of the line
        d = P2 - P1
        
        # Vector from P1 to P3
        v = P3 - P2
        
        # Projection of v onto d
        v_proj = (np.dot(v, d) / np.dot(d, d)) * d
        
        # Normal vector (perpendicular component)
        normal_vector = v - v_proj
        
        # Unit normal vector
       # unit_normal_vector = normal_vector / np.linalg.norm(normal_vector) if np.linalg.norm(normal_vector) != 0 else normal_vector
        print(normal_vector)
        return normal_vector

    def _create_homogeneous_matrix(self, base_point, point1, point2):
        """Create a homogeneous transformation matrix."""
        # Translation part
        T = np.eye(4)
        T[:3, 3] = base_point

        # Rotation part (assuming point1 and point2 define the x and y axes respectively)
        x_axis = np.array(point1) - np.array(base_point)
        x_axis /= np.linalg.norm(x_axis)

        y_axis = np.array(point2) - np.array(base_point)
        y_axis = y_axis - np.dot(y_axis, x_axis) * x_axis  # Make y_axis orthogonal to x_axis
        y_axis /= np.linalg.norm(y_axis)

        z_axis = np.cross(x_axis, y_axis)

        R = np.eye(4)
        R[:3, 0] = x_axis
        R[:3, 1] = y_axis
        R[:3, 2] = z_axis

        # Combine rotation and translation
        H = T @ R
        return H


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

def save_tracking_data(data, file_path):
    """
    Saves joint angles, camera extrinsics/intrinsics, and robot base transformation to a JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Calibration data saved to {file_path}")

def main(args):
    # Load the calibration data
    global rematch_markers
    rematch_markers = True
    
    blobs_dict : Dict[int,Blobs] = {}
    markers = None
    cams = load_camera_calib(path = "/home/aidara/augmented_imitation_learning/data_storage/calibration_output")
    for cam in cams: 
        path = "/home/aidara/augmented_imitation_learning/data_storage/" + args.project_name + f"/SVO_SN{cam.camera_id}.svo2"
        cam.init_zed(path) #f"cpp_multicam_rec/build/SVO_SN{cam.camera_id}.svo2")
        blobs_dict[cam.camera_id] = Blobs(cam.camera_id, NUM_BLOBS)
    
    frame_grabber = MultiCamSync(cams)
    print("Initialized cameras")
    
    robot_base_transform = np.eye(4)
     # We assume, that one camera is horizontal
    robot_rot_mat = cams[1].extrinsics[:3,:3]
    robot_base_transform[:3, :3] = robot_rot_mat.T  @ Rotation.from_euler('x', 90, degrees=True).as_matrix() 
    # rotate 90° around x axis, since this is somehow off.
    robot_base_transform[:3, 3] =  robot_base_transform[:3, :3] @np.array([500, 0, -500]) 
    
    robot = RobotSim(robot_base_transform=robot_base_transform,  visualization=False)
    print("Initialized virtual robot")   
    
    # Add the cameras and the robot base to the o3d visualizer
    if args.visuals:
        vis = open3d_visualizer(cams, robot_base_transform = robot_base_transform)
        hand_frame_vis = CoordFrameVis(vis)
        num_joints = robot.num_joints
        robot_frames_vis = CoordFrameVis(vis,num_coord_frame=num_joints,origin=robot_base_transform)
        print("Initialized Visualizer") 
    
    tracking_data = {
        "robot": {
            "base_transform": np.eye(4).tolist(),
            "states": [],
        },
        "cameras": [
            {
                "camera_id": cam.camera_id,
                "intrinsics": cam.intrinsics.tolist(),
                "extrinsics": (np.linalg.inv(robot_base_transform) @ np.linalg.inv(cam.extrinsics)).tolist()
            } for cam in cams
        ]
    }
    
    # Define the folder path
    folder_path = f"/home/aidara/augmented_imitation_learning/data_storage/{args.project_name}/raw_images"

    # Check if the folder exists, and create it if not
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    image_mat = sl.Mat()
    depth_mat = sl.Mat()
    xyz_mat = sl.Mat()
    loop = True
    
    timestep = 0
    
    while loop:
        img_list = []
        if frame_grabber.grab_frames() != sl.ERROR_CODE.SUCCESS:
            print("Failed to grab frames")
            loop = False
            break
        
        matched_blobs_uv = []
        
        for cam in cams:
            id = cam.camera_id
            cam.zed.retrieve_image(image_mat, sl.VIEW.LEFT)
            cam.zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            cam.zed.retrieve_measure(xyz_mat, sl.MEASURE.XYZ)
            #print(f"Cam {cam.camera_id} at timestamp {cam.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()}")
            img = image_mat.get_data()[:, :, :3] 
            img = img.astype(np.uint8)   
            img_list.append(img)
            xyz_data = xyz_mat.get_data()         
            transform = cam.extrinsics
            
            # save image as png to select the hsv values
            # check if raw_images folder exists
            
            

            # Save the image
            cv2.imwrite(f"{folder_path}/{id}_{timestep}.png", img)
                        
            #cv2.imwrite(f"home/aidara/augmented_imitation_learning/data_storage/{args.project_name}/raw_images/{id}_{timestep}.png", img)
            
            # Blob detection 
            blob_centers = blob_detection(img, id)
            
            if len(blob_centers) != 0:
                centers_as_int = []
                xyzs_global = []
                for center in blob_centers:
                    # Get the uv_point as integer
                    center_as_int = np.array(center, dtype=int)
                    centers_as_int.append(center_as_int)
                    
                    # Get the XYZ point in the global frame
                    xyz_stereo_point = xyz_data[center_as_int[1], center_as_int[0]]
                    xyz_stereo_point = np.array([xyz_stereo_point[0], xyz_stereo_point[1], xyz_stereo_point[2], 1])
                    direction = xyz_stereo_point[0:3] / np.linalg.norm(xyz_stereo_point[0:3])
                    xyz_stereo_point[0:3] = xyz_stereo_point[0:3] + 10 * direction     
                    xyz_global = np.dot(np.linalg.inv(transform), xyz_stereo_point)
                    xyzs_global.append(xyz_global[:3])
                
                # Visualize the points of depth measurements in the global frame
                #vis.visualize_points(xyzs_global, color = color_dict[id])
                
                # The blobs_dict acesses the Blobs_class that store the uv and xyz coordinates of the blobs 
                matched_uvs = blobs_dict[cam.camera_id].filter_blobs(centers_as_int, xyzs_global)
                
                for blob_id, matched_uv in enumerate(matched_uvs):
                    cv2.circle(img, (int(matched_uv[0]), int(matched_uv[1])), 5, blob_colors[blob_id], -1)
                    #cv2.putText(img, f"At point {matched_xyzs[blob_id]}", (int(matched_uv[0]) + 10 , int(matched_uv[1]) -10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                #uv.append(centers)
                if args.visuals:
                    cv2.imshow(f"Camera {id}", img)
               
        # Now we have consistent indices for the blobs
        # If the number of markers changes, we will set a flag to rematch the markers
        
        if markers is None:
            # This is just for the initialization 
            try: 
                markers = Markers(blobs_dict, cams) 
                keypoints = markers.keypoints
                if args.visuals:
                    vis.visualize_points(keypoints, color = (0, 0, 0))  
            except ValueError:
                print("Could not initialize markers")
        else:  
            store_state_dict = {}
            store_state_dict["id"] = timestep
            
            keypoints = markers.track_markers(blobs_dict)
            if args.visuals:
                vis.visualize_points(keypoints, color = [(255, 0, 0),(0, 255, 0),(0, 0, 255)])
            hand_coord_frame, point = markers.get_hand_pose()
            #vis.visualize_points([point], color = (0, 0, 0))
            if args.visuals:
                hand_frame_vis.update([hand_coord_frame])
            
            hand_coord_frame[:3, :3] = hand_coord_frame[:3, :3] @ Rotation.from_euler('y', 90, degrees=True).as_matrix()
            
            robot_goal_frame = np.linalg.inv(robot_base_transform) @ hand_coord_frame
            
            #calibration_data["robot"]["goal_poses"].append(robot_goal_frame.tolist())
            store_state_dict["goal_position"] = robot_goal_frame.tolist()
            
            robot.do_inverse_kinematics(hand_coord_frame)
            joint_transforms = robot.get_joint_transformations()
            if args.visuals:
                robot_frames_vis.update(joint_transforms)
            
            joint_angles = robot.get_joint_angles()
            #calibration_data["robot"]["joint_angles"].append(joint_angles)
            store_state_dict["joint_angles"] = joint_angles

            tracking_data["robot"]["states"].append(store_state_dict)

        timestep += 1
        
        cv2.waitKey(1)  # Allow OpenCV to update the window
    # Then get the uv coordinates from the image for the blob and call 
    robot.stop()
    print("Saving tracking data...")
    save_tracking_data(tracking_data, file_path= f"/home/aidara/augmented_imitation_learning/data_storage/{args.project_name}/tracking_data.json")
    print("done")
    

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Calibrate ZED cameras")
    parser.add_argument("--project_name", type=str, help="Name of the project")
    parser.add_argument("--visuals", action="store_true", help="Enable visualization")
    args = parser.parse_args()
    
    print("Doing the project: ", args.project_name)
    print("Visuals: ", args.visuals)
    main(args)   
    
    