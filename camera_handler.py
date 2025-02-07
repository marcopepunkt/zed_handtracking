########################################################################
#
# Copyright (c) 2020, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    Multi cameras sample showing how to open multiple ZED in one program
"""

import pyzed.sl as sl
import cv2
import numpy as np
import threading
import time
import signal
import blob_detector as bd
import open3d as o3d

zed_list = []
left_list = []
depth_list = []
timestamp_list = []
thread_list = []
stop_signal = False

NUM_POINTS = 2



def signal_handler(signal, frame):
    global stop_signal
    stop_signal=True
    time.sleep(0.5)
    exit()

def grab_run(index):
    global stop_signal
    global zed_list
    global timestamp_list
    global left_list
    global depth_list

    runtime = sl.RuntimeParameters()
    while not stop_signal:
        err = zed_list[index].grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            zed_list[index].retrieve_image(left_list[index], sl.VIEW.LEFT)
            zed_list[index].retrieve_measure(depth_list[index], sl.MEASURE.XYZ)
            timestamp_list[index] = zed_list[index].get_timestamp(sl.TIME_REFERENCE.CURRENT).data_ns
        time.sleep(0.001) #1ms
    zed_list[index].close()
	
def main():
    global stop_signal
    global zed_list
    global left_list
    global depth_list
    global timestamp_list
    global thread_list
    signal.signal(signal.SIGINT, signal_handler)

    print("Running...")
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30  # The framerate is lowered to avoid any USB3 bandwidth issues
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.MILLIMETER


    #List and open cameras
    name_list = []
    last_ts_list = []
    calibration_data = []
    cameras = sl.Camera.get_device_list()
    index = 0
    for cam in cameras:
        init.set_from_serial_number(cam.serial_number)
        name_list.append(str(cam.serial_number))
        calibration_data.append(bd.load_camera_calib(float(cam.serial_number))[0])
        print("Opening {}".format(name_list[index]))
        zed_list.append(sl.Camera())
        left_list.append(sl.Mat())
        depth_list.append(sl.Mat())
        timestamp_list.append(0)
        last_ts_list.append(0)
        status = zed_list[index].open(init)
        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            zed_list[index].close()
        index = index +1

    
    #Start camera threads
    for index in range(0, len(zed_list)):
        if zed_list[index].is_opened():
            thread_list.append(threading.Thread(target=grab_run, args=(index,)))
            thread_list[index].start()
            
    #Create windwos for open3d
    sphere, vis = bd.visualize_extrinsics(calibration_data, [[0,0,0], [0,0,0]])
    
    #Display camera images
    key = ''
    while key != 113:  # for 'q' key
        uv = []
        stereo_points = []
        # This is the main loop
        for index in range(0, len(zed_list)):
            if zed_list[index].is_opened():
                if (timestamp_list[index] > last_ts_list[index]):
                    img = left_list[index].get_data()
                    centers = bd.blob_detection(img)
                    if centers is None:
                        continue
                    # Do the blob detection here
                    for center in centers:
                        err, xyz_stereo = depth_list[index].get_value(int(center[0]), int(center[1]))
                        transform = calibration_data[index].extrinsics
                        xyz_stereo = np.array([xyz_stereo[0], xyz_stereo[1], xyz_stereo[2], 1])
                        xyz_global = np.dot(np.linalg.inv(transform), xyz_stereo)
                        stereo_points.append(xyz_global)
                        center_as_int = np.array(center, dtype=int)
                        cv2.circle(img, center_as_int, 5, (0, 0, 255), -1)   
                    uv.append(centers)
                                  
                    cv2.imshow(name_list[index], img)
                    last_ts_list[index] = timestamp_list[index]
        
        # now i have the uv coordinates from all cameras, i will now match them. First approach is stupid: 
        # Just see which uv coordinates are the closest to the previous frame.. 
        num_points = min(NUM_POINTS, len(stereo_points))
        for i in range(0, num_points):
            point = stereo_points[i]
            sphere[i].translate(point.reshape(-1)[:3], relative=False)
            vis.update_geometry(sphere[i])
            vis.poll_events()
            vis.update_renderer()
            
        # Do the triangulation here 
        # if len(uv) == len(zed_list) and len(uv) == 2 and uv[0] is not None and uv[1] is not None:
        #     #point = bd.triangulation(calibration_data, uv)
            
        #     # Update the visualization in open3d
        #     #vis.remove_geometry(sphere[0])
            
        #     #transform = np.eye(4)
        #     #transform[:3, 3] = point.reshape(-1)
        #     #sphere[0] = o3d.geometry.TriangleMesh.create_sphere(radius=10)
        #     sphere[0].translate(point.reshape(-1), relative=False)
        #     vis.update_geometry(sphere[0]) 
        #     #vis.add_geometry(sphere[0])
        #     vis.poll_events()
        #     vis.update_renderer() 

        #     print(point)
        # else: 
        #     print("Dot not detected in all cameras")
            
        key = cv2.waitKey(10)
                
    cv2.destroyAllWindows()

    #Stop the threads
    stop_signal = True
    for index in range(0, len(thread_list)):
        thread_list[index].join()

    print("\nFINISH")

if __name__ == "__main__":
    main()