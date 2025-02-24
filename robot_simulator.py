import pybullet as p
import time
import pybullet_data
from scipy.spatial.transform import Rotation
import numpy as np

class RobotSim:
    def __init__(self, robot_base_transform = np.identity(4), visualization = True):
        if visualization:
            physicsClient = p.connect(p.GUI)
        else: 
            physicsClient = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
        #p.setGravity(0,0,-10)
        #planeId = p.loadURDF("plane.urdf")
        base_pos, base_rot = self._transform_to_pos_quat(robot_base_transform)
        self.robot_id = p.loadURDF("franka_panda/panda.urdf",base_pos, base_rot, useFixedBase = True)
        self.end_effector_link_index = 11  # Index of the panda's end effector
        
        self.num_joints = p.getNumJoints(self.robot_id)
        self.movable_joints = [i for i in range(self.num_joints) if p.getJointInfo(self.robot_id, i)[2] != p.JOINT_FIXED]
        p.createConstraint(
            parentBodyUniqueId=self.robot_id,
            parentLinkIndex=-1,  # Base link
            childBodyUniqueId=-1,  # World frame
            childLinkIndex=-1,  # No specific link
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],  # No axis for fixed constraint
            parentFramePosition=[0, 0, 0],  # Base position
            childFramePosition=base_pos  # Attach to world at base_pos
        )
        
    def do_inverse_kinematics(self, target_pose):        
        target_position, target_orientation = self._transform_to_pos_quat(target_pose)

        # Compute desired joint positions using inverse kinematics
        joint_positions = p.calculateInverseKinematics(self.robot_id, self.end_effector_link_index, target_position, target_orientation)

        for desired_angle, joint in zip(joint_positions, self.movable_joints):
            p.resetJointState(self.robot_id, joint, desired_angle)
    
        # Step the simulation
        p.stepSimulation()
        
    
    def get_joint_transformations(self):
        """
        Get the homogeneous transformation matrices for all joints.
        Returns:
            A dictionary mapping joint indices to their 4x4 transformation matrices.
        """
        joint_transforms = []

        for joint in range(self.num_joints):
            # Get joint state (world position & orientation in quaternion form)
            link_state = p.getLinkState(self.robot_id, joint, computeForwardKinematics=True)
            pos = np.array(link_state[0])  # Position (x, y, z)
            quat = np.array(link_state[1])  # Quaternion (x, y, z, w)

            # Convert quaternion to a rotation matrix
            rotation_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)

            # Construct homogeneous transformation matrix
            T = np.eye(4)
            T[:3, :3] = rotation_matrix  # Set rotation part
            T[:3, 3] = pos*1000  # Set translation part

            joint_transforms.append(T)  # Store matrix

        return joint_transforms
    
    def get_joint_angles(self):
        """
        Get the joint angles of the robot.
        Returns:
            A list of joint angles in radians.
        """
        joint_angles = []
        for joint in self.movable_joints:
            joint_angles.append(p.getJointState(self.robot_id, joint)[0])

        return joint_angles
    
    def _transform_to_pos_quat(self, T):
        """
        Converts a homogeneous transformation matrix (4x4) into position (xyz) and quaternion (xyzw).
        """
        # Extract translation (position)
        position = T[:3, 3]/1000

        # Extract rotation matrix
        rotation_matrix = T[:3, :3]

        # Convert rotation matrix to quaternion (xyzw format)
        quaternion = Rotation.from_matrix(rotation_matrix).as_quat()

        return position, quaternion
    
    def stop(self):
        p.disconnect()

