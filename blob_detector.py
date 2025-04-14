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
from tracking_util import Hand, SphereMarker

NUM_BLOBS = 3
color_dict = {39725782: (221, 0, 252), 38580376: (0, 255, 238)}
blob_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

   
    





def save_tracking_data(data, file_path):
    """
    Saves joint angles, camera extrinsics/intrinsics, and robot base transformation to a JSON file.
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Calibration data saved to {file_path}")

def main(args):
    # Load the calibration data
    
    
    
    cams = load_camera_calib(path = "/home/aidara/augmented_imitation_learning/data_storage/calibration_output")
    for cam in cams: 
        path = "/home/aidara/augmented_imitation_learning/data_storage/" + args.project_name + f"/SVO_SN{cam.camera_id}.svo2"
        cam.init_zed(path) #f"cpp_multicam_rec/build/SVO_SN{cam.camera_id}.svo2")
    
    # This initializes the relevant parts of the Markertracking
    thumb = SphereMarker(cams,hsv_limits=[[44,100,45],[72,251,255]])
    index_base = SphereMarker(cams,hsv_limits=[[129,165,74],[255,255,255]])
    index_tip = SphereMarker(cams,hsv_limits=[[0,193,46],[42,255,255]])
    hand = None
    
    frame_grabber = MultiCamSync(cams)
    print("Initialized cameras")
    
    robot_base_transform = np.eye(4)
    
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
    
  
    loop = True
    
    timestep = 0
    
    
    while loop:
        timestep += 1

        # This is the frame grabber that synchronizes the cameras
        # It grabs the frames 
        if frame_grabber.grab_frames() != sl.ERROR_CODE.SUCCESS:
            print("Failed to grab frames, which usually means that the file ended")
            loop = False
            break
        
        for cam in cams:
            # This gets the image, depth and point cloud data and stores them in the camera object as np arrays
            cam.retrieve_data()
            
            # Save the image
            #cv2.imwrite(f"{folder_path}/{cam.camera_id}_{timestep}.png", cam.image)
            # Save the depth image
            #print(f"Saved Image to {folder_path}/{cam.camera_id}_{timestep}.png")      
            # if args.visuals:
            #     cv2.imshow(f"Camera {cam.camera_id}", cam.image)
                
        
        try:
            # This processes the camera images, does the blob detecction and the triangulation and gets a xyz position of the SphereMarker
            thumb.process_new_frame()
            index_base.process_new_frame()
            index_tip.process_new_frame()
            
            if args.visuals:
                for cam in cams:
                    np_image = cam.image.copy()
                    cv2.circle(np_image, thumb.processing_dict[cam]["center"], int(thumb.processing_dict[cam]["radius"]), (255, 0, 0), 2)  
                    cv2.circle(np_image, index_base.processing_dict[cam]["center"], int(index_base.processing_dict[cam]["radius"]), (0, 255, 0), 2)
                    cv2.circle(np_image, index_tip.processing_dict[cam]["center"], int(index_tip.processing_dict[cam]["radius"]), (0, 0, 255), 2)
                    cv2.imshow(f"Camera {cam.camera_id}",np_image)
                    cv2.waitKey(1)
                
        except ValueError as e:
            print(f"Could not get a position for all Sphere Markers: {e}")
            continue

            
            
        
        if hand is None:
            try: 
                hand = Hand(thumb, index_base, index_tip)
                print("Hand initialized")
            except ValueError:
                print("Hand initialization failed")
                continue
        else:
            # This updates the hand object with the new positions
            hand.track_hand()
            
            
        
        store_state_dict = {}
        store_state_dict["id"] = timestep

        
        
     
        hand_coord_frame, finger_dist = hand.get_hand_pose()
        if args.visuals:
            vis.visualize_points(hand.matrix, color = [(255, 0, 0),(0, 255, 0),(0, 0, 255)])
            hand_frame_vis.update([hand_coord_frame])
            
        store_state_dict["finger_distance"] = finger_dist

        hand_coord_frame[:3, :3] = hand_coord_frame[:3, :3] @ Rotation.from_euler('y', 90, degrees=True).as_matrix()
        
        robot_goal_frame = np.linalg.inv(robot_base_transform) @ hand_coord_frame
        
        #calibration_data["robot"]["goal_poses"].append(robot_goal_frame.tolist())
        store_state_dict["goal_position"] = robot_goal_frame.tolist()
        
        tracking_data["robot"]["states"].append(store_state_dict)
        
        
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
    
    