import cv2
import numpy as np
import pyzed.sl as sl
import sys
import os
import yaml
import open3d as o3d

origin_id = 39725782 # The serial number of the camera at the origin of the world frame

class camera_data:
    def __init__(self, camera_id, intrinsics, extrinsics):
        self.camera_id = camera_id
        self.intrinsics = np.array(intrinsics)
        self.extrinsics = np.array(extrinsics)
        
    def get_projection_matrix(self):
        return self.intrinsics @ self.extrinsics[:3, :]

def blob_detection(image_np):
    
   
    # Create a blob detector with default parameters
    color_lower = (0, 50, 160,0) # BGR format (This is for yellow color)
    color_upper = (70, 160, 255,255)   
    
    mask = cv2.inRange(image_np, color_lower, color_upper)
            
    #cv2.imshow("Mask", mask)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_NONE)
  
    if contours:
        blob = max(contours, key=cv2.contourArea)
    else:
        return None
    #print(cv2.contourArea(blob))
    if cv2.contourArea(blob) > 5:    
        M = cv2.moments(blob)
        center = [float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"])]
        #cv2.circle(image_np, center, 10, (0, 0, 255), -1)
        # Get the depth value at the center of the blob
        #depth_value = float(depth_np[center[1], center[0]])
        #cv2.putText(image_np, f"Distance: {depth_value} mm", (center[0] + 10, center[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        #print(f"Blob center at {center} has depth: {depth_value} mm")
        return center
    else:
        return None

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
                cams.append(camera_data(data["id"],data['intr_3x3'],data['extr_4x4']))
            except yaml.YAMLError as exc:
                print(exc)
    return cams



def visualize_extrinsics(cams, points):
    """Visualizes camera extrinsic matrices as coordinate frames with labels."""
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    

    for cam in cams:
        cam_id = cam.camera_id
        extr = cam.extrinsics
        extr = np.linalg.inv(extr)
        print(f"Camera {cam_id} at:\n{extr}")

        # Create frame and add to visualizer
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
        frame.transform(extr)
        vis.add_geometry(frame)
        
        text = o3d.t.geometry.TriangleMesh.create_text(text = f"Cam {cam_id}", depth = 5).to_legacy()
        text.paint_uniform_color([1, 0, 0])
        text.transform(extr)
        vis.add_geometry(text)
    
    o3d_points = []
        
    if points is not None:
        for point in points:
            point = np.array(point)
            transform = np.eye(4)
            transform[:3, 3] = point.reshape(-1)
            print(f"Point at:\n{point}")
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=10)
            sphere.transform(transform)
            vis.add_geometry(sphere)
            o3d_points.append(sphere)
            
    # Add origin frame
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20)
    frame.transform(np.eye(4))
    vis.add_geometry(frame)
    vis.poll_events()
    vis.update_renderer()
    # Run visualizer
    print("Done")
    #vis.destroy_window()
    return o3d_points, vis
        
        
if __name__ == "__main__":
    cams = load_camera_calib()
    
    points = []
    # Get the uv coordinates -> Must be normalized
    uv1 = [0.5*1280, 0.5*720]
    uv2 = [0.5*1280, 0.5*720]
    
    points.append(triangulation(cams, [uv1, uv2]))
    uv1 = [0.5*1280, 0.5*720]
    uv2 = [0.6*1280, 0.5*720]
    points.append(triangulation(cams, [uv1, uv2]))
    visualize_extrinsics(cams, points)
    
    
    # Then get the uv coordinates from the image for the blob and call 
    print("done")
    #main()
    