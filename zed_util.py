import pyzed.sl as sl

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

if __name__ == "__main__":
    zed = init_zed("calibration.svo")
    runtime = sl.RuntimeParameters()
    image = sl.Mat()
    