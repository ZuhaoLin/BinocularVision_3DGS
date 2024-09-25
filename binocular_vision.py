import nerfstudio.cameras.cameras as cameras
import numpy as np
import torch

X = torch.Tensor([1, 0, 0]).reshape((3, 1))
Y = torch.Tensor([0, 1, 0]).reshape((3, 1))
Z = torch.Tensor([0, 0, 1]).reshape((3, 1))

class binocular_eyes:
    def __init__(self, pd, w2o, cams=None):
        """
        Creates a pair of binocular eyes

        pd: pupilary distance. The distance between the two eyes from pupil to pupil
        """

        if cams is None:
            default = torch.FloatTensor([1])
            default = 1.0
            default_w2c = torch.eye(4)
            self.left_eye = cameras.Cameras(default_w2c, default, default, default, default, 600, 300)
            self.right_eye = cameras.Cameras(default_w2c, default, default, default, default, 600, 300)
        elif len(cams) == 2:
            self.left_eye = cams[0]
            self.right_eye = cams[1]
        elif isinstance(cams, cameras):
            self.left_eye, self.right_eye = cams, cams

        self.w2o = w2o

    def get_position(self):
        return self.w2o[:-1, -1]
    
    def get_rotation_matrix(self):
        return self.w2o[:-1, :-1]
    
    def set_position(self, position):
        self.w2o[:-1, -1] = position

    def set_rotation_matrix(self, rotation_matrix):
        self.w2o[:-1, :-1] = rotation_matrix