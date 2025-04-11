from typing import Dict, Optional


import cv2
import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


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
            print(f"Expected {self.num_blobs} blobs, but found {len(new_uv)}. Need to rematch markers")
            self.rematch_markers = True
            
        if len(self.uv) == 0:
            self.uv = np.array(new_uv)
            self.xyz_stereo_estimate = np.array(new_xyz)
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
 


class Markers: 
    def __init__(self, blob_dict, cams):
        self.cams = cams
        for cam_id, blobs in blob_dict.items():
            if blobs.len() != 3:
                print("For successful initialization all 3 Blobs must be seen from both cameras") 
                raise ValueError("Not all blobs are seen") 
        
        self.estimate = False 
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
        self.rematch_markers = False
        print("Initialized markers sucessfully")
        
    def track_markers(self, blob_dict): 
        """This function tracks the markers"""
        # If the marker number has not changed, we dont have to match the markers again
        
        for blob in blob_dict.values():
            if blob.rematch_markers:
                self.rematch_markers = True
        
        unmatched_dict = {}
        
        if self.rematch_markers:
            matched_pairs = self.match_blobs_to_markers(blob_dict)
        else: 
            matched_pairs = []
            
            for cam_id, blobs in blob_dict.items(): # For each camera
                # Load UV coordinates of the blobs
                uv = blobs.uv
                # Assign the blobs to one another to form th markers. 
                uv_indices = self.marker_id_dict[cam_id]            
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
        #return self.keypoints
    
    def match_blobs_to_markers(self, blobs_dict):
        """
        This function matches the blobs from all cameras and creates a dictionary that
        matches the blobs to one another e.g. 
        Blob 1 from camera 1 is matched to blob 2 from camera 2
        blob 2 from camera 1 is matched to blob 1 from camera 2
        blob 3 from camera 1 is matched to blob 3 from camera 2
        -> {2139430: [0, 1, 2], 1234235: [1, 0, 2]}
        """
        
        self.rematch_markers = False
        # 1) Gather camera 3D points
        xyz_cam = []
        uv_cam = []
        for cam in self.cams: # For each camera
            cam_id = cam.camera_id
            xyz_stereo_point = np.array(blobs_dict[cam_id].xyz_stereo_estimate)
            # Replace nan with 0 (if any)
            if np.isnan(xyz_stereo_point).any():
                print(f"Found NaN in the stereo estimate for camera {cam_id}")
                self.rematch_markers = True
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
            
        unmatched_indices = set(range(len(uv_cam[0]))) - set(self.marker_id_dict[self.cams[0].camera_id])
        if len(unmatched_indices) != 0:
            unmatched_dict[self.cams[0].camera_id] = unmatched_indices
        
        self.unmatched_dict = unmatched_dict    
        
        if len(self.unmatched_dict) != 0:
            print("Some markers are not seen in both cameras")
            self.estimate = True
        else:
            self.estimate = False
        
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
          
class Hand: 
    """This class is used to track the hand pose. It uses a kalman filter."""
    def __init__(self, markers :  Dict[tuple[int, int], Optional[Markers]] ):
        # For initialization, we need at least one fully initialized marker class
        valid_markers = [marker for marker in markers.values() if marker is not None]
        
        if len(valid_markers) == 0:
            raise ValueError("No valid markers found for initialization")

        
        # Make smth that is better 
        self.thumb = valid_markers[0].keypoints[0]
        self.index_top = valid_markers[0].keypoints[1]
        self.index_base = valid_markers[0].keypoints[2]
        
    @property
    def matrix(self):
        # This is to get the matrix of the hand pose
        return [self.thumb, self.index_base, self.index_top]

    @matrix.setter
    def matrix(self, new_values):
        # This is to set the matrix of the hand pose
        self.thumb, self.index_base, self.index_top = new_values

    def track_hand(self, markers: Dict[tuple[int, int], Optional["Markers"]]):
        from scipy.spatial.distance import cdist
        from scipy.optimize import linear_sum_assignment

        valid_markers = [marker for marker in markers.values() if marker is not None]
        assigned_markers = []
        for marker in valid_markers:
            if not marker.estimate:
                cost_matrix = cdist(self.matrix, marker.keypoints)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)

                updated_matrix = self.matrix.copy()
                for i, j in zip(row_ind, col_ind):
                    updated_matrix[i] = marker.keypoints[j]
                assigned_markers.append(updated_matrix)  # this updates thumb, index_base, index_top

        # now since we have the assigned markers, we just take the average of the assigned markers
        if len(assigned_markers) > 0:
            assigned_markers = np.array(assigned_markers)
            self.thumb = np.mean(assigned_markers[:, 0], axis=0)
            self.index_base = np.mean(assigned_markers[:, 1], axis=0)
            self.index_top = np.mean(assigned_markers[:, 2], axis=0)
        else:
            print("No valid markers found for tracking")
            return
        
    def get_hand_pose(self):
        base = np.mean([self.thumb, self.index_base], axis = 0)
        foward_vector = self._normal_to_line(self.thumb, self.index_base, self.index_top)
        forward_point = base + foward_vector
        finger_dist = np.linalg.norm(self.index_top - self.thumb)
        return self._create_homogeneous_matrix(base, foward_vector + base, self.thumb), finger_dist
    
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