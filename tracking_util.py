from typing import Dict, Optional


import cv2
import numpy as np

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


    
    

class SphereMarker:
    def __init__(self, cams, hsv_limits):
        self.cams = cams
        self.hsv_limits = hsv_limits
        self.pos = None
        self.processing_dict = {cam: {"center": (0,0), "radius": 0} for cam in self.cams}
        
        # Track previously used cameras for stability
        self.prev_cam1 = None
        self.prev_cam2 = None
        
        # Track camera performance over time
        self.cam_history = {cam: {'avg_radius': 0, 'count': 0} for cam in self.cams}
        
        # Consistency threshold - how many frames to consider before switching cameras
        self.consistency_threshold = 5
        
        # Camera switch counter - tracks how many consecutive frames a new camera has been better
        self.switch_counter = {}
        for cam in self.cams:
            self.switch_counter[cam] = 0
            
    
    def process_new_frame(self):
        # Store the previous position for stability
        prev_pos = self.pos
        
        # Process frames from all cameras
        for cam in self.cams:
            center, radius = self.blob_detection(cam)
            if center is not None:
                # Store the center and radius of the detected blob
                self.processing_dict[cam]["center"] = center
                self.processing_dict[cam]["radius"] = radius
                
                # Update camera history
                history = self.cam_history[cam]
                if history['count'] == 0:
                    history['avg_radius'] = radius
                else:
                    # Exponential moving average with 0.8 weight for new data
                    history['avg_radius'] = 0.8 * radius + 0.2 * history['avg_radius']
                history['count'] += 1
            else: 
                # If no blob is detected, set the radius to 0, so that the previous data is not lost, but has a low priority
                self.processing_dict[cam]["radius"] = 0
        
        # Ensure camera 39725782 is used if available and has a valid detection
        preferred_cam_id = 39725782
        preferred_cam = None
        
        # Find the preferred camera in our camera list
        for cam in self.cams:
            if cam.camera_id == preferred_cam_id:
                preferred_cam = cam
                break
        
        # Get all cameras with valid detections
        valid_cams = {cam: data for cam, data in self.processing_dict.items() if data["radius"] > 0}
        
        if len(valid_cams) < 2:
            # Not enough valid cameras for triangulation
            print("Not enough valid cameras for triangulation")
            return  # Keep the previous position
        
        # Determine which cameras to use
        if self.prev_cam1 is not None and self.prev_cam2 is not None:
            # Check if both previously used cameras are still valid
            if (self.prev_cam1 in valid_cams and self.prev_cam2 in valid_cams and 
                self.processing_dict[self.prev_cam1]["radius"] > 0 and 
                self.processing_dict[self.prev_cam2]["radius"] > 0):
                
                # Get the current top two cameras by radius
                top_two = sorted(valid_cams.items(), key=lambda x: x[1]["radius"], reverse=True)[:2]
                top_cam1, top_data1 = top_two[0]
                top_cam2, top_data2 = top_two[1]
                
                # Check if any of the top cameras is significantly better than our previous ones
                # and update switch counters
                for top_cam in [top_cam1, top_cam2]:
                    if top_cam not in [self.prev_cam1, self.prev_cam2]:
                        # If this top camera is not one of our previous cameras
                        if self.processing_dict[top_cam]["radius"] > 1.5 * min(
                            self.processing_dict[self.prev_cam1]["radius"],
                            self.processing_dict[self.prev_cam2]["radius"]):
                            # This camera is significantly better
                            self.switch_counter[top_cam] += 1
                        else:
                            # Reset counter if not significantly better
                            self.switch_counter[top_cam] = 0
                
                # Check if we should switch cameras
                switch_cam = None
                for cam, count in self.switch_counter.items():
                    if count >= self.consistency_threshold:
                        switch_cam = cam
                        break
                
                if switch_cam:
                    # Replace the camera with the lowest radius
                    if self.processing_dict[self.prev_cam1]["radius"] < self.processing_dict[self.prev_cam2]["radius"]:
                        self.prev_cam1 = switch_cam
                    else:
                        self.prev_cam2 = switch_cam
                    # Reset all counters after a switch
                    for cam in self.cams:
                        self.switch_counter[cam] = 0
                
                # Use the current selected cameras
                cam1, cam2 = self.prev_cam1, self.prev_cam2
                
            else:
                # One or both previous cameras are no longer valid, select new ones
                top_two = sorted(valid_cams.items(), key=lambda x: x[1]["radius"], reverse=True)[:2]
                cam1, data1 = top_two[0]
                cam2, data2 = top_two[1]
                self.prev_cam1, self.prev_cam2 = cam1, cam2
                # Reset switch counters
                for cam in self.cams:
                    self.switch_counter[cam] = 0
        else:
            # First run or reset, use the top two cameras
            # Prioritize preferred camera if available
            if preferred_cam and preferred_cam in valid_cams:
                cam1 = preferred_cam
                # Get the best camera that isn't the preferred one
                other_cams = {cam: data for cam, data in valid_cams.items() if cam != preferred_cam}
                if other_cams:
                    cam2, _ = sorted(other_cams.items(), key=lambda x: x[1]["radius"], reverse=True)[0]
                else:
                    # If no other valid cameras, just use the next best one (shouldn't happen with 2+ cameras)
                    other_cams = sorted(valid_cams.items(), key=lambda x: x[1]["radius"], reverse=True)
                    if len(other_cams) > 1:
                        cam2, _ = other_cams[1]
                    else:
                        return  # Not enough valid cameras
            else:
                # Preferred camera not available, use top two by radius
                top_two = sorted(valid_cams.items(), key=lambda x: x[1]["radius"], reverse=True)[:2]
                cam1, _ = top_two[0]
                cam2, _ = top_two[1]
            
            # Store the selected cameras
            self.prev_cam1, self.prev_cam2 = cam1, cam2
        
        # Get the data for the selected cameras
        data1 = self.processing_dict[cam1]
        data2 = self.processing_dict[cam2]
        
        # Calculate new position
        new_pos = self.triangulation(data1["center"], data2["center"], cam1, cam2)[0]
        
        # Apply smoothing if we have a previous position
        if prev_pos is not None:
            self.pos = 0.8 * new_pos + 0.2 * prev_pos
        else:
            self.pos = new_pos
            
            
            
    
    def blob_detection(self,cam):
        """This function detects blobs in an image and returns their centers"""
        
        image_np = cam.image.copy()
        image_np = cv2.GaussianBlur(image_np, (5, 5), 0)
        # If a field of interest is defined for this camera, 
        # create a mask by drawing a filled black rectangle outside the region
        # This helps exclude areas we don't want to process
        if hasattr(cam, 'field_of_interest') and cam.field_of_interest is not None:
            # Create a black mask the size of the image
            mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
            # Fill the region of interest with white
            cv2.rectangle(mask, cam.field_of_interest[0], cam.field_of_interest[1], 255, -1)
            # Apply the mask to keep only the region of interest
            image_np = cv2.bitwise_and(image_np, image_np, mask=mask)

        
        # Convert to HSV for easier color thresholding
        hsv = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)

        #cv2.imshow(f"HSV {cam_id}", hsv)
        # Define orange color range (tune these as need^ed)
        lower_hsv_limit = np.array(self.hsv_limits[cam.camera_id][0]) #np.array([5, 150, 123])
        upper_hsv_limit = np.array(self.hsv_limits[cam.camera_id][1]) #np.array([14, 255, 255])

        # Threshold to get only orange
        mask = cv2.inRange(hsv, lower_hsv_limit, upper_hsv_limit)
        
        
        #cv2.imshow(f"Theshhold Mask {cam.camera_id}", mask)

        # Clean up noise via opening/closing
        kernel = np.ones((3, 3), np.uint8)
        
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        #cv2.imshow(f"Cleaned Mask {cam.camera_id}", mask)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(image_np, contours, -1, (0, 255, 0), 2)
        
        center = None
        radius = 0
        
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            if radius > 3:
                center = (int(x), int(y))
                # Optional: draw the selected center
                #cv2.imshow(f"Center Mask {cam.camera_id}", image_np)
        return center, radius
    
    def triangulation(self, uv1,uv2, cam1, cam2):
        """This function returns the 3D position of a point in the global frame (from calibration pattern) from a single image"""
        assert len(uv1) == len(uv2), "The uv coordinates have the same length"
        #print(f"--- ES EXISTIEREN {len(uv[0])}")
        
        P1 = cam1.get_projection_matrix()
        P2 = cam2.get_projection_matrix()

        # Convert list of (u,v) to arrays of shape (2, N)
        pts1 = np.array(uv1, dtype=float).T  # shape (2, N)
        pts2 = np.array(uv2, dtype=float).T  # shape (2, N)
        
        X = cv2.triangulatePoints( P1, P2, pts1, pts2)
        # Remember to divide out the 4th row. Make it homogeneous
        X /= X[3]
        #print(X)
        #assert X.shape[1] == len(uv1), "The number of points must be the same"
        
        return X[:3].T
                    
class Hand: 
    """This class is used to track the hand pose. It uses a kalman filter."""
    def __init__(self, thumb, index_base, index_tip):
        if thumb is None or index_base is None or index_tip is None:
            raise ValueError("All markers must be initialized")
        
        if thumb.pos is None or index_base.pos is None or index_tip.pos is None:
            raise ValueError("Position not initialized")
        
        # For initialization, we need at least one fully initialized marker class
        # Make smth that is better 
        # self.thumb = thumb
        # self.index_tip = index_tip
        # self.index_base = index_base
        self.thumb_class = thumb
        self.index_tip_class = index_tip
        self.index_base_class = index_base
     
        
    @property
    def matrix(self):
        # This is to get the matrix of the hand pose
        return [self.thumb, self.index_base, self.index_tip]

    @matrix.setter
    def matrix(self, new_values):
        # This is to set the matrix of the hand pose
        self.thumb, self.index_base, self.index_tip = new_values

    
    @property
    def thumb(self):
        return self.thumb_class.pos

    @property
    def index_base(self):
        return self.index_base_class.pos

    @property
    def index_tip(self):
        return self.index_tip_class.pos
    
    def track_hand(self):
    
        
        pass
        
        # # now since we have the assigned markers, we just take the average of the assigned markers
        # if len(assigned_markers) > 0:
        #     assigned_markers = np.array(assigned_markers)
        #     self.thumb = np.mean(assigned_markers[:, 0], axis=0)
        #     self.index_base = np.mean(assigned_markers[:, 1], axis=0)
        #     self.index_top = np.mean(assigned_markers[:, 2], axis=0)
        # else:
        #     print("No valid markers found for tracking")
        #     return
        
    def get_hand_pose(self):
        base = np.mean([self.thumb, self.index_base], axis = 0)
        foward_vector = self._normal_to_line(self.thumb, self.index_base, self.index_tip)
        forward_point = base + foward_vector
        finger_dist = np.linalg.norm(self.index_tip - self.thumb)
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