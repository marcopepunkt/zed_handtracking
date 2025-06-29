import pyzed.sl as sl
import numpy as np
import time
import os
import yaml
import sys
import cv2

def init_zed(calib_path):
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30  # The framerate is lowered to avoid any USB3 bandwidth issues
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.MILLIMETER


    #List and open cameras
    zed_list = []
    name_list = []
    last_ts_list = []
    cameras = sl.Camera.get_device_list()
    index = 0
    for cam in cameras:
        init.set_from_serial_number(cam.serial_number)
        name_list.append("ZED {}".format(cam.serial_number))
        print("Opening {}".format(name_list[index]))
        cam = sl.Camera()
        _ = cam.open(init)
        zed_list.append(cam)
        time.sleep(1)
    
    return zed_list

class MultiCamSync:
    def __init__(self, cams):
        """
        cams: list of your camera objects (each with cam.zed)
        max_buffer_len: maximum frames stored in buffer
        """
        self.cams = cams
      
    def grab_frames(self):
        """
        Grab frames from all cameras, store in buffers as (timestamp, image_data).
        """
        
        if self.cams[0].zed.grab() != sl.ERROR_CODE.SUCCESS: return sl.ERROR_CODE.FAILURE
        timestamp = self.cams[0].zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
        
        for cam in self.cams[1:]:
            other_timestamp = cam.zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
            if other_timestamp > timestamp:
                #print("SKIPPED A FRAME")
                continue
            elif other_timestamp < timestamp - 33 - 33:
                if cam.zed.grab() != sl.ERROR_CODE.SUCCESS: return sl.ERROR_CODE.FAILURE
                if cam.zed.grab() != sl.ERROR_CODE.SUCCESS: return sl.ERROR_CODE.FAILURE     
                #print("SOME EXTRA GRABBING") 
            else:
                if cam.zed.grab() != sl.ERROR_CODE.SUCCESS: return sl.ERROR_CODE.FAILURE 
        
        
        return sl.ERROR_CODE.SUCCESS
    
    
class CameraData:
    def __init__(self, camera_id, intrinsics, extrinsics, distortion, d_fov, focal_length, h_fov, v_fov):
        self.camera_id = camera_id
        self.intrinsics = np.array(intrinsics)
        self.extrinsics = np.array(extrinsics)
        self.distortion = np.array(distortion)        
        self.d_fov = np.array(d_fov)
        self.focal_length = np.array(focal_length)
        self.h_fov = np.array(h_fov)
        self.v_fov = np.array(v_fov)
    
    def get_projection_matrix(self):
        # Suppose that the extrisics matrix is camera to world
        # T_wc = np.linalg.inv(self.extrinsics)
        # R_wc = T_wc[:3, :3]
        # t_wc = T_wc[:3, 3]
        # Rt = np.hstack((R_wc, t_wc.reshape(3, 1)))  
        return self.intrinsics @ np.linalg.inv(self.extrinsics)[:3, :4]

   
    def init_zed(self, svo_input_path):
            # loads the svo2 file and returns the frames of the depth and the left color image
        init_params = sl.InitParameters()
        init_params.set_from_svo_file(svo_input_path)
        init_params.svo_real_time_mode = False  # Don't convert in realtime
        init_params.coordinate_units = sl.UNIT.MILLIMETER  # Use milliliter units (for depth measurements)
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL 
        init_params.camera_resolution = sl.RESOLUTION.HD720
        
        print(f"The intrinsics are {init_params}")
                       
        zed = sl.Camera()

        # Open the SVO file specified as a parameter
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            sys.stdout.write(repr(err))
            zed.close()
            exit()
        cam_config = zed.get_camera_information().camera_configuration
        self.width = cam_config.resolution.width
        self.height = cam_config.resolution.height
        
        intr = cam_config.calibration_parameters.left_cam 
        # self.intrinsics = np.array([[intr.fx, 0, intr.cx], 
        #                             [0, intr.fy, intr.cy], 
        #                             [0, 0, 1]], dtype=np.float32)
        self.zed = zed
        zed.grab()
        
        self.image_mat = sl.Mat()
        self.depth_mat = sl.Mat()
        self.xyz_mat = sl.Mat()
        print(f"Camera {self.camera_id} initialized.")
        
        # # Save the first frame
        # self.retrieve_data()
        
        # Set empty_frame from {cam_id}_keyframe.jpg in the same folder as the SVO file
        svo_dir = os.path.dirname(svo_input_path)
        keyframe_path = os.path.join(svo_dir, f"{self.camera_id}_keyframe.jpg")
        if os.path.exists(keyframe_path):
            self.empty_frame = cv2.imread(keyframe_path)
        else:
            print(f"Warning: Keyframe not found at {keyframe_path}, using current image.")
            self.retrieve_data()
            self.empty_frame = self.image.copy()
        
    
    def retrieve_data(self):
        # Retrieve the left image and depth data
        self.zed.retrieve_image(self.image_mat, sl.VIEW.LEFT)
        self.zed.retrieve_measure(self.depth_mat, sl.MEASURE.DEPTH)
        self.zed.retrieve_measure(self.xyz_mat, sl.MEASURE.XYZRGBA)
        
        # Convert to numpy arrays
        img = self.image_mat.get_data()[:, :, :3] 
        self.image = img.astype(np.uint8)  
        self.depth = self.depth_mat.get_data()
        self.xyz = self.xyz_mat.get_data()
    
    
    def remove_human(self):
        """This function removes the human from the image"""
        if self.cube_color is None:
            raise ValueError("Cube color not specified")
        
        
        # Convert image to HSV for color-based segmentation
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        lower_bound = np.array(self.cube_color[0])
        upper_bound = np.array(self.cube_color[1])
        
        # Create mask for the cube
        cube_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Clean up the mask using morphological operations
        kernel = np.ones((5, 5), np.uint8)
        cube_mask = cv2.morphologyEx(cube_mask, cv2.MORPH_OPEN, kernel)
        cube_mask = cv2.morphologyEx(cube_mask, cv2.MORPH_CLOSE, kernel)
        
        # Create inverse mask for everything except the cube
        inverse_mask = cv2.bitwise_not(cube_mask)
        
        # Extract only the cube from the current frame
        cube_only = cv2.bitwise_and(self.image, self.image, mask=cube_mask)
        
        # Extract everything except the cube from the empty frame
        background = cv2.bitwise_and(self.empty_frame, self.empty_frame, mask=inverse_mask)
        
        self.cleaned_depth = cv2.bitwise_and(self.depth, self.depth, mask=cube_mask)
        # self.normalized_depth = cv2.normalize(self.depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        # self.normalized_cleaned_depth = cv2.normalize(self.cleaned_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        
        # cv2.imshow(f"Depth Cam {self.camera_id}", self.normalized_depth)
        # cv2.imshow(f"Cleaned Depth Cam {self.camera_id}", self.normalized_cleaned_depth)
        
        
        
        # Combine the cube with the background from empty frame
        self.cleaned_frame = cv2.add(cube_only, background)
        
        #return self.cleaned_frame
        
    
    
def load_camera_calib(path, id = None)-> list[CameraData]:
    """This function loads the camera calibration parameters from a yaml file. If an ID 
    :2#iterate over all files in the folder calibration_output
    is provided, it will only load the calibration parameters for that camera"""
    cams = []
    for idx,file in enumerate(os.listdir(path)):
        
        # Check if the file is a YAML file
        if not file.endswith(('.yaml', '.yml')):
            continue
        
        #open the file
        fullpath = os.path.join(path, file)
        with open(fullpath, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
                if id is not None and data["id"] != id:
                    continue
                #create a camera object
                cams.append(CameraData(data["id"],data['intr_3x3'],data['extr_4x4'],data['disto'], data['d_fov'], data['focal_length'], data['h_fov'], data['v_fov']))
            except yaml.YAMLError as exc:
                print(exc)
    return cams
    
    


if __name__ == "__main__":
    zed = init_zed("calibration.svo")
    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    