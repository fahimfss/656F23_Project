import numpy as np
import time
import cv2
import gymnasium
from gymnasium.spaces import Box
from collections import deque


class CarterReacherEnv(gymnasium.Env):
    def __init__(self, scene_path, seed=0, min_target_size=0.75, physics_dt=1.0/25.0, rendering_dt = 1.0/25.0, 
                 headless=True, image_stack=3, image_width=160, image_height=90, img_type='hwc', hf_img=(640, 360)):
        
        from omni.isaac.kit import SimulationApp
        self._simulation_app = SimulationApp({"headless": headless})

        from omni.isaac.core.utils.stage import open_stage
        open_stage(usd_path=scene_path)

        self.seed(seed)
        self._min_target_size = min_target_size

        self._image_width = image_width
        self._image_height = image_height

        self._img_type = img_type
        if self._img_type == 'chw':
            self._channel_axis = 0
            self._image_shape = (image_stack * 3, image_height, image_width)
        else:
            self._channel_axis = -1
            self._image_shape = (image_height, image_width, image_stack * 3)


        self._hf_img = hf_img

        self._image_buffer = deque([], maxlen=image_stack)

        self._proprioception_shape = (10,)
        self._v_w_low = np.array([-0.25, -0.4])
        self._v_w_high = np.array([0.4, 0.4])
        self._orientation_low = np.array([-1] * 3)
        self._orientation_high = np.array([1] * 3)
        self._pos_low = np.array([-5] * 5)
        self._pos_high = np.array([5] * 5)


        self._lower = np.array([44, 98, 65])
        self._upper = np.array([97, 255, 128])
        
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
        from omni.isaac.sensor import Camera
        from omni.isaac.core.utils.rotations import euler_angles_to_quat
        from omni.isaac.core.utils.viewports import set_camera_view
        from omni.isaac.core.prims.rigid_prim import RigidPrim
        from omni.isaac.core.world import World
        
        self._world = World(stage_units_in_meters=1.0, physics_dt=physics_dt, rendering_dt=rendering_dt)

        self._headless = headless

        if not headless:
            set_camera_view(
                eye=[2.5, 9, 17], target=[0, 0.5, 3.09072], 
                camera_prim_path="/OmniverseKit_Persp")

        carter_p, target_p = self.get_randon_poses()

        self._carter = self._world.scene.add(
            Articulation(prim_path="/World/carter_v2", name="my_carter", 
            position=np.array([carter_p[0], carter_p[1], 3.09072]),
            orientation=euler_angles_to_quat([0, 0, carter_p[2]], degrees=True))
        )
        self._controller = DifferentialController(name="simple_control", wheel_radius=0.04295, wheel_base=0.4132)
        self._robot_wheels = ["joint_wheel_left", "joint_wheel_right"]

        self._camera = Camera(
            prim_path="/World/carter_v2/chassis_link/stereo_cam_right/stereo_cam_right_sensor_frame/camera_sensor_right",
            resolution=(hf_img[0], hf_img[1])
        )

        self._target_prim = RigidPrim(
            name="target",
            position=np.array([target_p[0], target_p[1], 3.590713]),
            orientation=euler_angles_to_quat([0, 0, target_p[2]], degrees=True),
            prim_path="/World/Cylinder",
            scale=np.array([0.5, 0.5, 1.0])
        )
        self._target = self._world.scene.add(self._target_prim)

        self._need_reset = True

        
    def reset(self):
        if not self._headless:
            from omni.isaac.core.utils.viewports import set_camera_view
            set_camera_view(
                eye=[2.5, 9, 17], target=[0, 0.5, 3.09072], 
                camera_prim_path="/OmniverseKit_Persp")

        carter_p, target_p = self.get_randon_poses()

        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.utils.rotations import euler_angles_to_quat

        self._world.scene.remove_object("my_carter", True)
        self._carter = self._world.scene.add(
            Articulation(prim_path="/World/carter_v2", name="my_carter", 
            position=np.array([carter_p[0], carter_p[1], 3.09072]),
            orientation=euler_angles_to_quat([0, 0, carter_p[2]], degrees=True))
        )

        self._world.reset()

        self._target_prim.set_world_pose(
            position=np.array([target_p[0], target_p[1], 3.590713]),
            orientation=euler_angles_to_quat([0, 0, target_p[2]], degrees=True)
        )

        self._controller.reset()
        self._camera.initialize()

        img = None
        for i in range(5):
            self._world.step(render=True)
            img = self._camera.get_rgb()
        
        hf_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(hf_img, (self._image_width, self._image_height))

        if self._img_type == 'chw':
            img = np.transpose(img, [2, 0, 1])
        for _ in range(self._image_buffer.maxlen):
            self._image_buffer.append(img)
        self._latest_image = np.concatenate(self._image_buffer, 
                                            axis=self._channel_axis)
        self._latest_proprioception = np.array([0, 0], dtype=np.float32)
        last_action = np.array([0, 0], dtype=np.float32)
        self._need_reset = False

        from omni.isaac.core.utils.rotations import quat_to_euler_angles
        orientation = quat_to_euler_angles(self._carter._articulation_view.get_world_poses()[1][0], True)
        orientation = orientation / 180
        position = self._carter._articulation_view.get_world_poses()[0][0] 
        proprioception = np.concatenate((last_action, orientation, position, self._target_pos))

        if not self._headless:
            from omni.isaac.core.utils.viewports import set_camera_view
            set_camera_view(
                eye=[2.5, 9, 17], target=[0, 0.5, 3.09072], 
                camera_prim_path="/OmniverseKit_Persp")

        return (hf_img, self._latest_image, proprioception)


    def step(self, action):
        assert not self._need_reset

        v, w = action[0], action[1]

        wheel_dof_indices = [self._carter.get_dof_index(
            self._robot_wheels[i]) for i in range(len(self._robot_wheels))]
        actions = self._controller.forward(command=[v, w])
        from omni.isaac.core.utils.types import ArticulationAction
        joint_actions = ArticulationAction()
        joint_actions.joint_velocities = np.zeros(self._carter.num_dof)
        if actions.joint_velocities is not None:
            for j in range(len(wheel_dof_indices)):
                joint_actions.joint_velocities[wheel_dof_indices[j]] = actions.joint_velocities[j]
        self._carter.apply_action(joint_actions)

        self._world.step(render=True)

        img = self._camera.get_rgb() 
        hf_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(hf_img, (self._image_width, self._image_height))

        target_size = self.get_target_size(img)
        reward = target_size -1

        if self._img_type == 'chw':
            img = np.transpose(img, [2, 0, 1])
        
        self._image_buffer.append(img)
        self._latest_image = np.concatenate(self._image_buffer,
                                            axis=self._channel_axis)
        
        from omni.isaac.core.utils.rotations import quat_to_euler_angles
        orientation = quat_to_euler_angles(self._carter._articulation_view.get_world_poses()[1][0], True)
        position = self._carter._articulation_view.get_world_poses()[0][0]

        done = False

        if abs(orientation[0]) > 30 or abs(orientation[1]) > 30:
            done = True
            reward = -10
            self._need_reset = True

        if not done and target_size >= self._min_target_size:
            done = True
            self._need_reset = True

            if not self._headless:
                from omni.isaac.core.utils.viewports import set_camera_view
                set_camera_view(
                    eye=[2.5, 9, 17], target=[0, 0.5, 3.09072], 
                    camera_prim_path="/OmniverseKit_Persp")

        orientation = orientation / 180
        proprioception = np.concatenate((action, orientation, position, self._target_pos))

        return (hf_img, self._latest_image, proprioception), reward, done, {}


    def get_randon_poses(self):
        # part = np.random.choice((1, 2))
        # if part == 1:
        carter_x = 4.2
        carter_y = 0.2  # np.random.uniform(low=-1.75, high=3.25)
        # else:
        #     carter_x = np.random.uniform(low=-1, high=-3.5)
        #     carter_y = np.random.uniform(low=-1.75, high=3.25)

        part = np.random.choice((1, 2))
        if part == 1:
            target_x = np.random.uniform(low=1.0, high=3.25)
            target_y = np.random.uniform(low=-1.75, high=3.25)
        else:
            target_x = np.random.uniform(low=-1, high=-3.5)
            target_y = np.random.uniform(low=-1.75, high=3.25)

        dist = np.linalg.norm(np.array([carter_x, carter_y])-np.array([target_x, target_y]))

        if dist > 0.5:
            or1 = 180 # np.random.uniform(low=-179, high=179)
            or2 = np.random.uniform(low=-179, high=179)

            self._target_pos = np.array([target_x, target_y])
            return (carter_x, carter_y, or1), (target_x, target_y, or2)
        
        else: 
            return self.get_randon_poses()

    
    def get_target_size(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self._lower, self._upper)
        output = cv2.bitwise_and(img,img, mask= mask)
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(blackAndWhiteImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.fillPoly(blackAndWhiteImage, pts=contours, color=(255, 255, 255))
        target_size = np.sum(blackAndWhiteImage/255.) / blackAndWhiteImage.size

        return target_size

    def close(self):
        self._simulation_app.close()
        cv2.destroyAllWindows()


    @property
    def image_space(self):
        return Box(low=0, high=255, shape=self._image_shape)

    @property
    def proprioception_space(self):
        low = np.concatenate((self._v_w_low, self._orientation_low, self._pos_low))
        high = np.concatenate((self._v_w_high, self._orientation_high, self._pos_high))
        return Box(low=low, high=high)

    @property
    def observation_space(self):
        return self.proprioception_space
    
    @property
    def action_space(self):
        return Box(low=self._v_w_low, high=self._v_w_high)

    def seed(self, seed=None):
        self.np_random, seed = gymnasium.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]