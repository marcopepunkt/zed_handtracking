import open3d as o3d
import numpy as np
from collections import deque
import copy

class open3d_visualizer:
    def __init__(self, cams, robot_base_transform, max_trail_length=50):
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
        text = o3d.t.geometry.TriangleMesh.create_text(
            text="Origin", depth=5
        ).to_legacy()
        text.paint_uniform_color([1, 0, 0])
        self.vis.add_geometry(text)
        
        # Add the camera base 
        base = o3d.geometry.TriangleMesh.create_coordinate_frame(size=20)
        base.transform(robot_base_transform)
        self.vis.add_geometry(base)
        text = o3d.t.geometry.TriangleMesh.create_text(
            text="Robot Base", depth=5
        ).to_legacy()
        text.paint_uniform_color([1, 0, 0])
        text.transform(robot_base_transform)
        self.vis.add_geometry(text)
        
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

        #view_control = self.vis.get_view_control()
        #param = view_control.convert_to_pinhole_camera_parameters()
        #o3d.io.write_pinhole_camera_parameters("viewpoint.json", param)
        
                
        # Update visualization
        self.vis.update_geometry(self.point_cloud)
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        """Closes the visualization window."""
        self.vis.destroy_window()

class CoordFrameVis:
    def __init__(self, vis, num_coord_frame = 1, size=20, origin = np.eye(4)):
        """
        A class for managing a triangle mesh (coordinate frame) representing a hand.
        
        Parameters:
            vis: Instance of open3d_visualizer
            size: Size of the coordinate frame
        """
        self.vis = vis  # Reference to the visualizer
        self.size = size
        self.coord_frames = []
        self.origin = origin
        for _ in range(num_coord_frame):
            joint_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=self.size)
            
            self.coord_frames.append([copy.deepcopy(np.asarray(joint_frame.vertices)),joint_frame])
            self.vis.vis.add_geometry(joint_frame)
      
        
        ## Store the initial vertices to allow direct transformation
        #self.original_vertices = copy.deepcopy(np.asarray(self.hand_frame.vertices))

    def update(self, transform_matrices):
        """
        Updates the hand frame position by transforming the existing mesh instead of replacing it.
        
        Parameters:
            transform_matrix: 4x4 transformation matrix to set the new hand frame position.
        """
        assert len(transform_matrices) == len(self.coord_frames), "Number of transformation matrices must match the number of coordinate frames"
        
        for transform_matrix, coord_frame in zip(transform_matrices, self.coord_frames):
            original_vertices, frame = coord_frame[0], coord_frame[1]
            # Shift the origin
            transform_matrix=  transform_matrix
            # Transform the original vertices
            transformed_vertices = (transform_matrix[:3, :3] @ original_vertices.T).T + transform_matrix[:3, 3]

            # Update the hand frame geometry directly
            frame.vertices = o3d.utility.Vector3dVector(transformed_vertices)
            frame.compute_vertex_normals()
            self.vis.vis.update_geometry(frame)
            
        self.vis.vis.poll_events()
        self.vis.vis.update_renderer()