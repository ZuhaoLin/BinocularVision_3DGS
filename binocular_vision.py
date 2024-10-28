import nerfstudio.cameras.cameras as cameras
import nerfstudio.cameras.camera_utils as camera_utils
import utils
import torch
import copy
import numpy as np

from jaxtyping import Float, Int, Shaped
from torch import Tensor
from typing import Dict, List, Literal, Optional, Tuple, Union
from nerfstudio.cameras.cameras import CameraType

X = torch.Tensor([1, 0, 0]).reshape((3, 1))
Y = torch.Tensor([0, 1, 0]).reshape((3, 1))
Z = torch.Tensor([0, 0, 1]).reshape((3, 1))
UP = -Z

class binocular_eyes:
    def __init__(self, pd, w2o, cams=None):
        """
        Creates a pair of binocular eyes

        pd: pupilary distance. The distance between the two eyes from pupil to pupil
        """

        self.pd = pd

        if cams is None:
            default = torch.FloatTensor([1])
            default = 1.0
            default_c2w = torch.eye(4)
            self.width = 600
            self.height = 300
            self.left_eye = eyeball(
                default_c2w,
                default,
                default,
                default,
                default,
                self.width,
                self.height
            )
            self.right_eye = eyeball(
                default_c2w,
                default,
                default,
                default,
                default,
                self.width,
                self.height
            )
        elif isinstance(cams, cameras.Cameras):
            self.width = cams.width
            self.height = cams.height
            self.left_eye = eyeball(
                cams.camera_to_worlds,
                cams.fx,
                cams.fy,
                cams.cx,
                cams.cy,
                cams.width,
                cams.height,
                cams.distortion_params,
                cams.camera_type,
                cams.times,
                cams.metadata
            )
            self.right_eye = eyeball(
                cams.camera_to_worlds,
                cams.fx,
                cams.fy,
                cams.cx,
                cams.cy,
                cams.width,
                cams.height,
                cams.distortion_params,
                cams.camera_type,
                cams.times,
                cams.metadata
            )
        elif isinstance(cams, eyeball):
            self.width = cams.width
            self.height = cams.height
            self.left_eye, self.right_eye = copy.deepcopy(cams), copy.deepcopy(cams)
        elif len(cams) == 2:
            # Note: Work on this later, make sure it is an eyeball and not cam obj
            self.left_eye = cams[0]
            self.right_eye = cams[1]

        self.w2o = w2o
        
        self._init_eye_positions()

    def _init_eye_positions(self):
        c2w_left, c2w_right = torch.eye(4), torch.eye(4)                                            # Default identity
        o2w = utils.quick_viewmat_inv(self.w2o)
        c2w_left, c2w_right = copy.deepcopy(o2w), copy.deepcopy(o2w)                                # Assume eyes start with same orientation as object
        c2w_left[0, -1] -= self.pd/2                                                                # Get the positions of the cameras
        c2w_right[0, -1] += self.pd/2

        self.left_eye.camera_to_worlds = c2w_left
        self.right_eye.camera_to_worlds = c2w_right

    def face_lookat(self, lookat_point: torch.Tensor):
        '''
        Set the "face" to look at a certain point
        '''
        lookat_point = lookat_point.reshape(3, 1)

        position = self.get_position()
        lookat = lookat_point - position
        o2w = torch.eye(4)
        o2w[:-1, :] = camera_utils.viewmatrix(
            lookat.flatten(),
            UP.flatten(),
            position.flatten()
        )

        self.w2o = utils.quick_viewmat_inv(o2w)
        self._set_eye_positions()

    
    def eye_lookat(self, lookat_point: torch.Tensor):
        '''
        Set both 'eyes' to look at a certain point
        '''
        c2w_left = self.left_eye.camera_to_worlds                                                   # Get rot and translation in world frame
        c2w_right = self.right_eye.camera_to_worlds
        
        left_lookat = lookat_point.reshape(3, 1) - c2w_left[:3, -1][:, None]                        # lookat vector, from eye to point
        right_lookat = lookat_point.reshape(3, 1) - c2w_right[:3, -1][:, None]

        left_viewmatrix = torch.eye(4)
        right_viewmatrix = torch.eye(4)

        left_viewmatrix[:-1, :] = camera_utils.viewmatrix(
            left_lookat.flatten(),
            UP.flatten(),
            c2w_left[:3, -1]
        )                                                                                           # Viewmatrix (camera to world)
        right_viewmatrix[:-1, :] = camera_utils.viewmatrix(
            right_lookat.flatten(),
            UP.flatten(),
            c2w_right[:3, -1]
        )

        self.left_eye.camera_to_worlds = left_viewmatrix                                            # C2W matrices
        self.right_eye.camera_to_worlds = right_viewmatrix

    def yaw(self, x):
        '''
        Rotate around the y axis (up), changing yaw
        '''
        Ry = utils.get_Ry(x, shape=4, as_type=torch.FloatTensor)
        self.w2o = Ry @ self.w2o
        self.left_eye.yaw(x)
        self.right_eye.yaw(x)

        self._set_eye_positions()

    def pitch(self, x):
        Rx = utils.get_Rx(x, shape=4, as_type=torch.FloatTensor)
        self.w2o = Rx @ self.w2o
        self.left_eye.yaw(x)
        self.right_eye.yaw(x)

        self._set_eye_positions()

    def move_forward(self, x):
        '''
        Move the binocular eyes forward
        '''
        self.w2o[2, -1] -= x                                                                             # Increase z by x (forward)
        self._set_eye_positions()

    def move_backwards(self, x):
        self.move_forward(-x)

    def move_right(self, x):
        self.w2o[0, -1] -= x
        self._set_eye_positions()
    
    def move_left(self, x):
        self.move_right(-x)

    def move_up(self, x):
        self.w2o[1, -1] += x
        self._set_eye_positions()

    def move_down(self, x):
        self.move_up(-x)
    
    def get_left_eye_position(self):
        return self.left_eye.camera_to_worlds[:-1, -1][:, None]
    
    def get_right_eye_position(self):
        return self.right_eye.camera_to_worlds[:-1, -1][:, None]

    def get_position(self):
        return utils.quick_viewmat_inv(self.w2o)[:-1, -1][:, None]
    
    def get_rotation_matrix(self):
        return self.w2o[:-1, :-1]
    
    def get_eyes_w2c(self):
        return (self.get_left_eye_w2c(), self.get_right_eye_w2c())

    def get_left_eye_w2c(self):
        return utils.quick_viewmat_inv(self.left_eye.camera_to_worlds)
    
    def get_right_eye_w2c(self):
        return utils.quick_viewmat_inv(self.right_eye.camera_to_worlds)
    
    def get_eyes_c2w(self):
        return (self.left_eye.camera_to_worlds, self.right_eye.camera_to_worlds)
    
    def get_intrinsics(self):
        Ks1 = self.left_eye.get_intrinsics_matrices().reshape((1, 3, 3)).float()
        Ks2 = self.right_eye.get_intrinsics_matrices().reshape((1, 3, 3)).float()
        Ks = torch.cat((Ks1, Ks2), 0).float()
        return Ks
    
    def _set_eye_positions(self):
        temp_left = copy.deepcopy(self.w2o)
        temp_right = copy.deepcopy(self.w2o)
        temp_left[0, -1] += self.pd/2
        temp_right[0, -1] -= self.pd/2

        left_position = utils.quick_viewmat_inv(temp_left)[:-1, -1]
        right_position = utils.quick_viewmat_inv(temp_right)[:-1, -1]

        self.left_eye.camera_to_worlds[:-1, -1] = left_position.flatten()                     # Convert back to cam to world matrices
        self.right_eye.camera_to_worlds[:-1, -1] = right_position.flatten()

    def set_position(self, position: torch.Tensor):
        '''
        Set the position for the 'face' in global coordinates
        '''
        position = position.reshape(3, 1)
        o2w = utils.quick_viewmat_inv(self.w2o)
        o2w[:-1, -1] = position.flatten()                                                      # Position of 'face'
        self.w2o = utils.quick_viewmat_inv(o2w)
        self._set_eye_positions()

class eyeball(cameras.Cameras):
    def __init__(
        self,
        camera_to_worlds: Float[Tensor, "*batch_c2ws 3 4"],
        fx: Union[Float[Tensor, "*batch_fxs 1"], float],
        fy: Union[Float[Tensor, "*batch_fys 1"], float],
        cx: Union[Float[Tensor, "*batch_cxs 1"], float],
        cy: Union[Float[Tensor, "*batch_cys 1"], float],
        width: Optional[Union[Shaped[Tensor, "*batch_ws 1"], int]] = None,
        height: Optional[Union[Shaped[Tensor, "*batch_hs 1"], int]] = None,
        distortion_params: Optional[Float[Tensor, "*batch_dist_params 6"]] = None,
        camera_type: Union[
            Int[Tensor, "*batch_cam_types 1"],
            int,
            List[CameraType],
            CameraType,
        ] = CameraType.PERSPECTIVE,
        times: Optional[Float[Tensor, "num_cameras"]] = None,
        metadata: Optional[Dict] = None,
    ) -> None:

        # Init eyeball the same way as cameras
        super().__init__(
            camera_to_worlds,
            fx,
            fy,
            cx,
            cy,
            width,
            height,
            distortion_params,
            camera_type,
            times,
            metadata
        )

    def yaw(self, x):
        Ry = utils.get_Ry(x, shape=4, as_type=torch.FloatTensor)
        w2c = self.get_w2c()
        self.camera_to_worlds = utils.quick_viewmat_inv(Ry @ w2c)

    def generate_yaw_peeks(self, x, mat_type='c2w'):
        Ry = utils.get_Ry(x, shape=4, as_type=torch.FloatTensor)
        w2c = self.get_w2c()
        w2c = Ry @ w2c
        if str.lower(mat_type)=='c2w':
            c2w = utils.quick_multiviewmat_inv(w2c)
            return c2w
        elif str.lower(mat_type)=='w2c':
            return w2c
        else:
            raise ValueError('mat_type must be c2w or 2wc')

    def pitch(self, x):
        Rx = utils.get_Rx(x, shape=4, as_type=torch.FloatTensor)
        w2c = self.get_w2c()
        self.camera_to_worlds = utils.quick_viewmat_inv(Rx @ w2c)

    def roll(self, x):
        Rz = utils.get_Rz(x, shape=4, as_type=torch.FloatTensor)
        w2c = self.get_w2c()
        self.camera_to_worlds = utils.quick_viewmat_inv(Rz @ w2c)

    def get_w2c(self):
        return utils.quick_viewmat_inv(self.camera_to_worlds)