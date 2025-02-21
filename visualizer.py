import open3d as o3d
import numpy as np
from collections import deque
import copy

class open3d_visualizer:
    def __init__(self, cams, max_trail_length=50):
        """Initializes the visualizer with the cameras."""
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        
        self.point_cloud = o3d.geometry.PointCloud()

        # Fixed-length buffer for points and colors
        self.trail_points = deque(maxlen=max_trail_length)
        self.trail_colors = deque(maxlen=max_trail_length)

        # Initialize the geometry (empty at first)
        self.point_cloud.points = o3d.utility.Vector3dVector(np.empty((0, 3)))
        self.point_cloud.colors = o3d.utility.Vector3dVector(np.empty((0, 3)))
        self.vis.add_geometry(self.point_cloud)
        
        # Visualize each camera extrinsic
        for cam in cams:
            cam_id = cam.camera_id
            extr = np.linalg.inv(cam.extrinsics)
            print(f"Camera {cam_id} at:\n{extr}")

            # Camera coordinate frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=50)
            frame.transform(extr)
            self.vis.add_geometry(frame)
            
            # Example text label (optional - depends on your Open3D version)
            text = o3d.t.geometry.TriangleMesh.create_text(
                text=f"Cam {cam_id}", depth=5
            ).to_legacy()
            text.paint_uniform_color([1, 0, 0])
            text.transform(extr)
            self.vis.add_geometry(text)
        
        # A global origin frame
        global_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20)
        self.vis.add_geometry(global_frame)

        self.vis.poll_events()
        self.vis.update_renderer()

    def visualize_points(self, new_points, color=[1.0, 0.0, 0.0]):
        """
        Appends new_points (each with 'color') to the trail and updates the view.

        If you want one color per point, pass an array of colors with the same
        length as 'new_points'. Otherwise, a single color is used for all.
        """
        new_points = np.asarray(new_points, dtype=float)

        # Handle both single color [R,G,B] or a list of colors
        color = np.asarray(color, dtype=float)
        if color.ndim == 1:  # single color => replicate for each new point
            color = np.tile(color, (len(new_points), 1))

        # Extend the deque by each new point/color
        for pt, col in zip(new_points, color):
            self.trail_points.append(pt)
            self.trail_colors.append(col)

        # Convert to NumPy arrays for Open3D
        points_array = np.array(self.trail_points)
        colors_array = np.array(self.trail_colors)

        self.point_cloud.points = o3d.utility.Vector3dVector(points_array)
        self.point_cloud.colors = o3d.utility.Vector3dVector(colors_array)

        # Update visualization
        self.vis.update_geometry(self.point_cloud)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        """Closes the visualization window."""
        self.vis.destroy_window()

class HandFrame:
    def __init__(self, vis, size=20):
        """
        A class for managing a triangle mesh (coordinate frame) representing a hand.
        
        Parameters:
            vis: Instance of open3d_visualizer
            size: Size of the coordinate frame
        """
        self.vis = vis  # Reference to the visualizer
        self.size = size
        self.hand_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.size)
        self.vis.vis.add_geometry(self.hand_frame)  # Add the initial hand frame
        
        # Store the initial vertices to allow direct transformation
        self.original_vertices = copy.deepcopy(np.asarray(self.hand_frame.vertices))

    def update(self, transform_matrix):
        """
        Updates the hand frame position by transforming the existing mesh instead of replacing it.
        
        Parameters:
            transform_matrix: 4x4 transformation matrix to set the new hand frame position.
        """
        # Transform the original vertices
        transformed_vertices = (transform_matrix[:3, :3] @ self.original_vertices.T).T + transform_matrix[:3, 3]

        # Update the hand frame geometry directly
        self.hand_frame.vertices = o3d.utility.Vector3dVector(transformed_vertices)

        # Ensure Open3D detects changes
        self.hand_frame.compute_vertex_normals()
        
        # Update renderer without resetting the camera view
        self.vis.vis.update_geometry(self.hand_frame)
        self.vis.vis.poll_events()
        self.vis.vis.update_renderer()