from typing import Dict, Optional
import json
import argparse
import os
from itertools import combinations

import cv2
import numpy as np
import pyzed.sl as sl

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.spatial.transform import Rotation

from visualizer import open3d_visualizer, CoordFrameVis
from zed_util import MultiCamSync, CameraData, load_camera_calib
from robot_simulator import RobotSim
from tracking_util import Blobs, Markers, Hand

NUM_BLOBS = 3
color_dict = {39725782: (221, 0, 252), 38580376: (0, 255, 238)}
blob_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

   
    



def blob_detection(image_np, cam_id):
    """This function detects blobs in an image and returns their centers"""
    
    if cam_id == 33137761:
        bounding_box_robot_base = ((1530,300), (1800,700)) # Top left and bottom right corners
    elif cam_id == 36829049:
            bounding_box_robot_base = ((199,560), (343,866)) # Top left and bottom right corners
    elif cam_id == 39725782:
        bounding_box_robot_base = ((987,331), (1176,482)) # Top left and bottom right corners
    else:
        raise ValueError("Base cutout is not definied for this camera")
    
    image_np = cv2.rectangle(image_np, bounding_box_robot_base[0], bounding_box_robot_base[1], (0, 0, 0), -1)

    
    # Convert to HSV for easier color thresholding
    hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

    #cv2.imshow(f"HSV {cam_id}", hsv)
    
    # Define orange color range (tune these as needed)
    lower_orange = np.array([5, 150, 123])
    upper_orange = np.array([14, 255, 255])

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

    cv2.drawContours(image_np, contours, -1, (0, 255, 0), 2)
    
    centers = []
    for cnt in contours:
        # Approx. the sphere with a minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius > 5:  # filter out too-small circles
            centers.append((int(x), int(y)))
   
    if len(centers) > 3: 
        cv2.waitKey(1000)
        print(f"Camera {cam_id} detected more than 3 blobs")
        #raise ValueError("Detected more than 3 blobs")    
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
    
    blobs_dict : Dict[int,Blobs] = {}
    markers = None
    hand = None
    
    
    cams = load_camera_calib(path = "/home/aidara/augmented_imitation_learning/data_storage/calibration_output")
    for cam in cams: 
        path = "/home/aidara/augmented_imitation_learning/data_storage/" + args.project_name + f"/SVO_SN{cam.camera_id}.svo2"
        cam.init_zed(path) #f"cpp_multicam_rec/build/SVO_SN{cam.camera_id}.svo2")
        blobs_dict[cam.camera_id] = Blobs(cam.camera_id, NUM_BLOBS)
        
    marker_dict: Dict[tuple[int, int], Optional[Markers]] = {}
    for cam1, cam2 in combinations(cams, 2):
        marker_dict[(cam1.camera_id, cam2.camera_id)] = None  # Explicitly allow None as a value
    print("Initialized the markers dict")
    
    frame_grabber = MultiCamSync(cams)
    print("Initialized cameras")
    
    robot_base_transform = np.eye(4)
     # We assume, that one camera is horizontal
    # robot_rot_mat = cams[1].extrinsics[:3,:3]
    # robot_base_transform[:3, :3] = robot_rot_mat.T  @ Rotation.from_euler('x', 90, degrees=True).as_matrix() 
    # # rotate 90° around x axis, since this is somehow off.
    # robot_base_transform[:3, 3] =  robot_base_transform[:3, :3] @np.array([500, 0, -500]) 
    
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
    active_marker = None
    
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
            #print(f"Cam {id}")
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
            

            # Save the image
            cv2.imwrite(f"{folder_path}/{id}_{timestep}.png", img)
                        
           
                
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
                # Filtering the blobs will sort the blobs in the same order as the previous timestep
                blobs_dict[cam.camera_id].filter_blobs(centers_as_int, xyzs_global)

                #for blob_id, matched_uv in enumerate(blobs_dict[cam.camera_id].uv):
                    #cv2.circle(img, (int(matched_uv[0]), int(matched_uv[1])), 5, blob_colors[blob_id], -1)
                    #cv2.putText(img, f"At point {matched_xyzs[blob_id]}", (int(matched_uv[0]) + 10 , int(matched_uv[1]) -10 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                #uv.append(centers)
            if args.visuals:
                cv2.imshow(f"Camera {id}", img)
               
        # Now we have consistent indices for the blobs
        # If the number of markers changes, we will set a flag to rematch the markers
        
        
        for (cam1_id, cam2_id), marker in marker_dict.items():        
            relevant_blobs_dict = {cam1_id: blobs_dict[cam1_id], cam2_id: blobs_dict[cam2_id]}
            relevant_cams = [cam for cam in cams if cam.camera_id in relevant_blobs_dict.keys()]
            if marker is None:
                try:         
                    # Try to initialize 
                    marker = Markers(relevant_blobs_dict, relevant_cams) 
                    marker_dict[(cam1_id, cam2_id)] = marker
                    # If initialization successfull, it is not an estimate and therefore accurate. 
                    #curr_marker = marker
                except ValueError:
                    print(f"Could not initialize markers between cameras {cam1_id} and {cam2_id}")
                    continue
            else: 
                #                  
                marker.track_markers(relevant_blobs_dict)
        
        if hand is None:
            try: 
                hand = Hand(marker_dict)
                print("Hand initialized")
            except ValueError:
                print("Hand initialization failed")
                continue
        else:
            hand.track_hand(marker_dict)
            
            
        
        
        # # We have a marker and it is now just an estimate
        # if curr_marker.estimate:
        #     print("Current marker is an estimate. Find better one.")
        #     # If there is a better marker configuration, we will update the curr_marker
        #     if any(marker is not None and not marker.estimate for marker in marker_dict.values()):
        #         # Find the first non-estimate marker
        #         for (cam1_id, cam2_id), marker in marker_dict.items():
        #             if marker is not None and not marker.estimate:
        #                 print(f"Found a better marker configuration between cameras {cam1_id} and {cam2_id}")

        #                 # now we have to align the marker with the previous one
        #                 # We will use the Hungarian algorithm to find the best match
        #                 # Get the keypoints of the current marker
        #                 keypoints = curr_marker.keypoints
        #                 # Get the keypoints of the new marker
        #                 new_keypoints = marker.keypoints
        #                 # Compute the cost matrix
        #                 cost_matrix = cdist(keypoints, new_keypoints)
        #                 # Solve the assignment problem
        #                 row_ind, col_ind = linear_sum_assignment(cost_matrix)
                        
        #                 marker.reorder_markers(col_ind)
        #                 # Get the new keypoints
        #                 keypoints = marker.keypoints
        #                 # Get the keypoints of the new marker                  
        #                 curr_marker = marker
        #                 break
        #     else: # If there isnt a better marker configuration, we will take the average of the keypoints where depth is not NaN
        #         print("No better marker configuration found")
        #         break 
        #         raise ValueError("No valid marker configuration found")
        #         # Take the average of valid keypoints
        #         all_keypoints = []
        #         for marker in marker_dict.values():
        #             if marker is not None:
        #                 valid_keypoints = [kp for kp in marker.keypoints if not np.isnan(kp.depth)]
        #                 all_keypoints.extend(valid_keypoints)

        #         if all_keypoints:
        #             keypoints = np.mean(all_keypoints, axis=0)
        # else: 
        #     keypoints = curr_marker.keypoints  
                 
        # if keypoints is None or len(keypoints) == 0 or np.isnan(keypoints).any():
        #     print("No keypoints available")
        #     cv2.waitKey(1)
        #     continue
        
        store_state_dict = {}
        store_state_dict["id"] = timestep

        
        
        if args.visuals:
            vis.visualize_points(hand.matrix, color = [(255, 0, 0),(0, 255, 0),(0, 0, 255)])
        hand_coord_frame, finger_dist = hand.get_hand_pose()
        
        store_state_dict["finger_distance"] = finger_dist
        
        if args.visuals:
            hand_frame_vis.update([hand_coord_frame])
        
        hand_coord_frame[:3, :3] = hand_coord_frame[:3, :3] @ Rotation.from_euler('y', 90, degrees=True).as_matrix()
        
        robot_goal_frame = np.linalg.inv(robot_base_transform) @ hand_coord_frame
        
        #calibration_data["robot"]["goal_poses"].append(robot_goal_frame.tolist())
        store_state_dict["goal_position"] = robot_goal_frame.tolist()
        
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
    
    