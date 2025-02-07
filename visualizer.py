import open3d as o3d
import numpy as np
from collections import deque

class open3d_visualizer:
    def __init__(self, cams, max_trail_length=50):
        """Initializes the visualizer with the cameras."""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        
        self.point_cloud = o3d.geometry.PointCloud()
        
        initial_value = np.array([0., 0., 0.])
        self.trail_points = deque([initial_value.copy() for _ in range(max_trail_length)], maxlen=max_trail_length)  # Fixed-length buffer
        self.point_cloud.points = o3d.utility.Vector3dVector(self.trail_points)
        self.vis.add_geometry(self.point_cloud)
        
        for cam in cams:
            cam_id = cam.camera_id
            extr = cam.extrinsics
            extr = np.linalg.inv(extr)
            print(f"Camera {cam_id} at:\n{extr}")

            # Create frame and add to visualizer
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
            frame.transform(extr)
            self.vis.add_geometry(frame)
            
            text = o3d.t.geometry.TriangleMesh.create_text(text = f"Cam {cam_id}", depth = 5).to_legacy()
            text.paint_uniform_color([1, 0, 0])
            text.transform(extr)
            self.vis.add_geometry(text)
        
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20)
        frame.transform(np.eye(4))
        self.vis.add_geometry(frame)
        self.vis.poll_events()
        self.vis.update_renderer()
            

    def visualize_points(self, new_points):
        """Updates the visualization with the points."""
        self.trail_points.extend(new_points)  # Add new point, old ones auto-remove
        points_array = np.array(self.trail_points)
        self.point_cloud.points = o3d.utility.Vector3dVector(points_array)
        self.vis.update_geometry(self.point_cloud)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        """Closes the visualization window."""
        self.vis.destroy_window()