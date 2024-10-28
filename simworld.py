import torch
import numpy as np
from gsplat.rendering import rasterization

class simworld:
    def __init__(self, datalib: dict, device='cuda'):

        self.means = torch.from_numpy(datalib['position']).float().to(device)
        self.quats = torch.from_numpy(datalib['rot']).float().to(device)
        self.scales = torch.from_numpy(datalib['scales']).float().to(device)
        self.opacities = torch.from_numpy(datalib['opacity']).float().to(device)
        self.colors = torch.from_numpy(datalib['color']).float().to(device)

    def render(self, viewmats, cam_intrinsics, width, height, device='cuda') -> torch.Tensor:
        render_colors, render_alphas, meta = rasterization(
            self.means.float().to(device),
            self.quats.float().to(device),
            self.scales.float().to(device),
            self.opacities.float().to(device),
            self.colors.float().to(device),
            viewmats.float().to(device),
            cam_intrinsics.float().to(device),
            width=width,
            height=height,
            near_plane=0,
            render_mode='RGB+D'
         )
      
        C = render_colors.shape[0]
        assert render_colors.shape == (C, height, width, 4)
        assert render_alphas.shape == (C, height, width, 1)

        render_rgbs = render_colors[..., 0:3]
        render_depths = render_colors[..., 3:4]
        render_depths = render_depths / render_depths.max()

        rgbs = torch.clamp(render_rgbs, 0.0, 1.0).reshape(C, height, width, 3).cpu()
        img = (rgbs*255).type(torch.IntTensor)

        return img