import cv2
import numpy as np
import argparse
import yaml 

import pyzed.sl as sl

from zed_util import init_zed

chessboard_size = (11, 8)  # Change this to your chessboard's size
square_size = 30  # Define the real-world size of a square in mm

def main(args):
    zed_list = init_zed(args.calib_svo_path)
    zed_serials = [zed.get_camera_information().serial_number for zed in zed_list]
    intrinsics = [zed.get_camera_information().camera_configuration.calibration_parameters.left_cam for zed in zed_list]
    intrinsics = [{
    'fx': intr.fx, 
    'fy': intr.fy,
    'cx': intr.cx,
    'cy': intr.cy,
    } for intr in intrinsics]
    
    disto = [zed.get_camera_information().camera_configuration.calibration_parameters.left_cam.disto for zed in zed_list]
        
    # create intrinsics matrix
    intrinsics_mat = []
    for intr in intrinsics:
        fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']
        intrinsics_mat.append(np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32))

    # get the frames from all initialized cameras 
    images = [sl.Mat() for _ in zed_list]
    images_np = []
    runtime = sl.RuntimeParameters()
    for idx, zed in enumerate(zed_list):
        err = zed.grab(runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            print(f"Error grabbing images from ZED camera {zed_serials[idx]}")
            exit()
        
        zed.retrieve_image(images[idx], sl.VIEW.LEFT)
        image_np = images[idx].get_data()[:, :, :-1]  ## leave out alpha
        images_np.append(image_np)
    

    for cam_idx, image in enumerate(images_np):
        # must be reset at each iteration to avoid error
        objp = np.zeros((np.prod(chessboard_size), 3), np.float32)
        objp[:, :2] = np.indices(chessboard_size).T.reshape(-1, 2)
        objp *= square_size

        objpoints = []  # 3d points in real-world space
        imgpoints = []  # 2d points in image plane.  
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
        """cv2.imshow("Image", gray)
        cv2.waitKey()
        cv2.destroyAllWindows()"""
        # Get the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        if ret: # if corners are found
            objpoints.append(objp)
            # Refine the corner positions
            criteria = (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
        else:
            print("Chessboard corners not found.")
            cv2.imshow("Projected points", image)
            cv2.waitKey()
            exit()

        """ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], intrinsics_mat[cam_idx], disto[cam_idx],
                                                           flags=cv2.CALIB_USE_INTRINSIC_GUESS+
                                                                cv2.CALIB_FIX_INTRINSIC +
                                                                cv2.CALIB_FIX_PRINCIPAL_POINT+cv2.CALIB_FIX_FOCAL_LENGTH)"""
        ret, rvec, tvec = cv2.solvePnP(objpoints[0], imgpoints[0], intrinsics_mat[cam_idx], disto[cam_idx], None, None, useExtrinsicGuess=False, flags=cv2.SOLVEPNP_ITERATIVE)
        if ret: # if calibration is successful and we found the extrinsics
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            extr_4x4 = np.eye(4)
            extr_4x4[:3, :3] = rotation_matrix
            extr_4x4[:3, 3] = tvec.T
            
            # Visualize the calibration results and project the points into the image
            imgpoints2, _ = cv2.projectPoints(objpoints[0], rvec, tvec, intrinsics_mat[cam_idx], disto[cam_idx])
            error = cv2.norm(imgpoints[0], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            print(f"Total reprojection error: {error}")

            objpoints[0] += np.array([1, 1, 0]) * square_size
            points_proj, _ = cv2.projectPoints(objpoints[0], rvec, tvec, intrinsics_mat[cam_idx], disto[cam_idx])
            print("points_proj succeeded")

            # create cv2 mat from np array
            img = np.ascontiguousarray(image, dtype=np.uint8)
            cv2.drawChessboardCorners(img, chessboard_size, points_proj, True)
            cv2.drawFrameAxes(img, intrinsics_mat[cam_idx], disto[cam_idx], rvec, tvec, 300)
            cv2.imshow("Projected points", img)
            cv2.waitKey()
            cv2.destroyAllWindows()
            
            with open(rf"{args.calib_output_path}/extr_{cam_idx}_{zed_serials[cam_idx]}.yml", 'w') as file:
                yaml.dump({"id": zed_serials[cam_idx], 'extr_4x4': extr_4x4.tolist(), "intr_3x3": intrinsics_mat[cam_idx].tolist(), "disto": disto[cam_idx].tolist()}, file)

            print("Translation vector:\n", tvec)
        else:
            print("Camera calibration failed.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Args for camera calibration")
    # Must calibrate for each m -(machine) folder separately
    parser.add_argument('--calib_svo_path', type=str, default=None, help="Path to folder with calibration SVO files.")
    parser.add_argument('--calib_output_path', type=str, default=None, help="Output folder for calibration files.")

    args = parser.parse_args()
    
    main(args)