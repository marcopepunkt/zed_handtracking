from typing import Dict, Optional, List, Tuple
import json
import argparse
import os
import csv
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

def load_segments_from_csv(csv_path: str) -> List[Tuple[int, int]]:
    """
    Load video segments from a CSV file.
    The CSV format is expected to have columns for segment name, start frame, end frame,
    and possibly additional metadata.
    
    Args:
        csv_path: Path to the CSV file containing segments
        
    Returns:
        List of tuples (start_timestamp, end_timestamp)
    """
    segments = []
    try:
        with open(csv_path, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            # Skip header row
            header = next(csv_reader, None)
            print(f"CSV header: {header}")
            
            for row in csv_reader:
                if len(row) >= 3:  # At least segment name, start, end
                    try:
                        # Assuming format: [segment_name, start_frame, end_frame, ...]
                        segment_name = row[0]
                        start_timestamp = int(row[1])
                        end_timestamp = int(row[2])
                        print(f"Parsed segment: {segment_name}, frames {start_timestamp} to {end_timestamp}")
                        segments.append((start_timestamp, end_timestamp))
                    except ValueError as e:
                        print(f"Error parsing row {row}: {e}")
                else:
                    print(f"Skipping row with insufficient data: {row}")
    except Exception as e:
        print(f"Error loading CSV file {csv_path}: {e}")
    
    return segments

def get_cam_config(args, cams):
    hsv_limits = {
        "thumb": {},
        "index_base": {},
        "index_tip": {}
    }

    # Check if cam_config in args.calib_path exists
    if not os.path.exists(os.path.join(args.calib_path, "cam_config.json")):
        print("No cam_config.json found in the calibration path. Using default HSV values.")
    
        for cam in cams:
            cam.field_of_interest = None
            cam.cube_color = None
            hsv_limits["thumb"][cam.camera_id] = [[44,100,45],[72,251,255]]
            hsv_limits["index_base"][cam.camera_id] = [[129,165,74],[255,255,255]]
            hsv_limits["index_tip"][cam.camera_id] = [[0,193,46],[42,255,255]]
    
    else: 
        cam_config = json.load(open(os.path.join(args.calib_path, "cam_config.json")))
        
        for cam in cams:
            hsv_limits["thumb"][cam.camera_id] = cam_config[str(cam.camera_id)]["hsv_limits_thumb"]
            hsv_limits["index_base"][cam.camera_id] = cam_config[str(cam.camera_id)]["hsv_limits_index_base"]
            hsv_limits["index_tip"][cam.camera_id] = cam_config[str(cam.camera_id)]["hsv_limits_index_tip"]
            cam.cube_color = cam_config[str(cam.camera_id)]["cube_color"]
            cam.field_of_interest = cam_config[str(cam.camera_id)]["field_of_interest"]
    return hsv_limits

def process_segment(args, cams : List[CameraData], segment_start: int, segment_end: int, segment_id: int):
    """
    Process a single video segment with the given start and end timestamps.
    
    Args:
        args: Command line arguments
        cams: List of camera objects
        segment_start: Start timestamp of the segment
        segment_end: End timestamp of the segment
        segment_id: ID of the current segment
    
    """
    
 
    hsv_limits = get_cam_config(args, cams)
        
    
    thumb = SphereMarker(cams, hsv_limits["thumb"])
    index_base = SphereMarker(cams, hsv_limits["index_base"])
    index_tip = SphereMarker(cams, hsv_limits["index_tip"])
    hand = None
    
    frame_grabber = MultiCamSync(cams)
    print(f"Processing segment {segment_id}: frames {segment_start} to {segment_end}")
    
    robot_base_transform = np.eye(4)
    
    robot = RobotSim(robot_base_transform=robot_base_transform, visualization=False)
    
    # Add the cameras and the robot base to the o3d visualizer
    vis = None
    hand_frame_vis = None
    robot_frames_vis = None
    if args.visuals:
        vis = open3d_visualizer(cams, robot_base_transform=robot_base_transform)
        hand_frame_vis = CoordFrameVis(vis)
        num_joints = robot.num_joints
        robot_frames_vis = CoordFrameVis(vis, num_coord_frame=num_joints, origin=robot_base_transform)
        print("Initialized Visualizer") 
    
    tracking_data = {
        "segment_id": segment_id,
        "start_timestamp": segment_start,
        "end_timestamp": segment_end,
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
    
    # Define the folder path for this segment
    og_name = os.path.basename(args.original_path)
    segment_folder = f"/home/aidara/augmented_imitation_learning/data_storage/{args.project_name}/{og_name}_Episode_{segment_id}"
    raw_images_folder = f"{segment_folder}/raw_images"

    # Check if the folders exist, and create them if not
    for folder in [segment_folder, raw_images_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    loop = True
    timestep = 0
    current_frame = segment_start
    
    # Directly set the SVO position to the segment start frame
    print(f"Setting SVO position to frame {segment_start}...")
    for cam in cams:
        cam.zed.set_svo_position(segment_start) 
    
    # Grab the first frame at the segment start position
    if loop and frame_grabber.grab_frames() != sl.ERROR_CODE.SUCCESS:
        print("Failed to grab frames at segment start position")
        loop = False
    
    print(f"Starting processing at frame {current_frame}")
    
    # Process frames within the segment range
    while loop and current_frame <= segment_end:
        
        
        
        # Grab synchronized frames from all cameras
        if frame_grabber.grab_frames() != sl.ERROR_CODE.SUCCESS:
            print("Failed to grab frames, which usually means that the file ended")
            loop = False
            break
        
        for cam in cams:
            # This gets the image, depth and point cloud data and stores them in the camera object as np arrays
            cam.retrieve_data()
            cam.remove_human()
            cv2.imwrite(f"{raw_images_folder}/{cam.camera_id}_raw.png", cam.image)
            
            # Save the image
            cv2.imwrite(f"{raw_images_folder}/{cam.camera_id}_{current_frame}.png", cam.cleaned_frame)
            #cv2.imwrite(f"{raw_images_folder}/{cam.camera_id}_{current_frame}_depth.tiff", cam.cleaned_depth)
            
            if args.visuals:
                cv2.imshow(f"Camera {cam.camera_id}", cam.cleaned_frame)
                
        try:
            # This processes the camera images, does the blob detection and the triangulation and gets a xyz position of the SphereMarker
            thumb.process_new_frame()
            index_base.process_new_frame()
            index_tip.process_new_frame()
            
            if args.visuals:
                for cam in cams:
                    np_image = cam.cleaned_frame.copy()
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
        store_state_dict["frame"] = current_frame
     
        hand_coord_frame, finger_dist = hand.get_hand_pose()
        if args.visuals and vis is not None:
            vis.visualize_points(hand.matrix, color = [(255, 0, 0),(0, 255, 0),(0, 0, 255)])
            hand_frame_vis.update([hand_coord_frame])
            
        store_state_dict["finger_distance"] = finger_dist

        hand_coord_frame[:3, :3] = hand_coord_frame[:3, :3] @ Rotation.from_euler('y', 90, degrees=True).as_matrix()
        
        robot_goal_frame = np.linalg.inv(robot_base_transform) @ hand_coord_frame
        
        store_state_dict["goal_position"] = robot_goal_frame.tolist()
        
        tracking_data["robot"]["states"].append(store_state_dict)
        
        print(f"Processed frame {current_frame}")
        current_frame += 1
        timestep += 1
        
        cv2.waitKey(1)  # Allow OpenCV to update the window
    
    robot.stop()
    print(f"Finished processing segment {segment_id}")
    save_tracking_data(tracking_data, file_path=f"{segment_folder}/tracking_data.json")

def main(args):
    # Load the calibration data
    cams = load_camera_calib(path=args.calib_path)
    
    # Initialize ZED cameras with SVO files
    for cam in cams: 
        path = args.original_path + f"/SVO_SN{cam.camera_id}.svo2"
        cam.init_zed(path)
        
    print("Initialized cameras")
    
    # Load segments from CSV file
    csv_path = f"{args.original_path}/segments.csv"
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}. Creating a default segment for the entire video.")
        # If no CSV file, process the entire video as one segment
        process_segment(args, cams, 0, float('inf'), 0)
    else:
        segments = load_segments_from_csv(csv_path)
        if not segments:
            print("No valid segments found in the CSV file. Processing entire video as one segment.")
            process_segment(args, cams, 0, float('inf'), 0)
        else:
            print(f"Found {len(segments)} segments to process")
            # Process each segment
            for i, (start, end) in enumerate(segments):
                # Reset cameras to the beginning for each segment
                if i > 0:
                    for cam in cams:
                        cam.zed.close()
                        path = f"{args.original_path}/SVO_SN{cam.camera_id}.svo2"
                        cam.init_zed(path)
                
                process_segment(args, cams, start, end, i)
    
    print("All segments processed successfully")
    

if __name__ == "__main__":  
    parser = argparse.ArgumentParser(description="Calibrate ZED cameras")
    parser.add_argument("--project_name", type=str, help="Name of the project")
    parser.add_argument("--visuals", action="store_true", help="Enable visualization")
    parser.add_argument("--original_path", type=str, help="Path to the csv files and svo files")
    parser.add_argument("--calib_path", type=str, help="Path to the calibration files")
    args = parser.parse_args()
    
    print("Doing the project: ", args.project_name)
    print("Visuals: ", args.visuals)
    
    if args.original_path is None:
        raise ValueError("Original path must be specified")
    if args.calib_path is None:
        raise ValueError("Calibration path must be specified")
    if args.project_name is None:
        raise ValueError("Project name must be specified")
    
    main(args)   
    
    