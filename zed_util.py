import pyzed.sl as sl
import numpy as np
import time

def init_zed(calib_path):
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 15  # The framerate is lowered to avoid any USB3 bandwidth issues
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
                print("SKIPPED A FRAME")
                continue
            elif other_timestamp < timestamp - 33 - 33:
                if cam.zed.grab() != sl.ERROR_CODE.SUCCESS: return sl.ERROR_CODE.FAILURE
                if cam.zed.grab() != sl.ERROR_CODE.SUCCESS: return sl.ERROR_CODE.FAILURE     
                print("SOME EXTRA GRABBING") 
            else:
                if cam.zed.grab() != sl.ERROR_CODE.SUCCESS: return sl.ERROR_CODE.FAILURE 
            
        return sl.ERROR_CODE.SUCCESS
    
    
class CameraData:
    def __init__(self, camera_id, intrinsics, extrinsics):
        self.camera_id = camera_id
        self.intrinsics = None
        self.extrinsics = np.array(extrinsics)
        
    def get_projection_matrix(self):
        world_to_camera = np.linalg.inv(self.extrinsics)
        return self.intrinsics @ world_to_camera[:3, :]

   
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
        cam_config = zed.get_camera_information().camera_configuration
        self.width = cam_config.resolution.width
        self.height = cam_config.resolution.height
        
        intr = cam_config.calibration_parameters.left_cam 
        fx, fy, cx, cy = intr.fx, intr.fy, intr.cx, intr.cy
        self.intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            
        self.zed = zed
    


if __name__ == "__main__":
    zed = init_zed("calibration.svo")
    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    