import torch
import numpy as np
from gsplat.rendering import rasterization
import utils

class simworld:
    def __init__(self, datalib: dict, device='cuda'):

        self.means = torch.from_numpy(datalib['position']).float().to(device)
        self.quats = torch.from_numpy(datalib['rot']).float().to(device)
        self.scales = torch.from_numpy(datalib['scales']).float().to(device)
        self.opacities = torch.from_numpy(datalib['opacity']).float().to(device)
        self.colors = torch.from_numpy(datalib['color']).float().to(device)

        self.added_ind = None

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
    
    def add_splats(
            self,
            means: torch.Tensor,
            quats: torch.Tensor,
            scales: torch.Tensor,
            opacities: torch.Tensor,
            colors: torch.Tensor
    ):
        means = means.reshape(-1, 3)
        quats = quats.reshape(-1, 4)
        scales = scales.reshape(-1, 3)
        colors = colors.reshape(-1, self.colors.shape[1])

        N = means.shape[0]

        if not torch.all(
            torch.Tensor([
                means.shape[0],
                quats.shape[0],
                scales.shape[0],
                *opacities.size(),
                colors.shape[0],
            ])
            == N
        ):
            raise ValueError('Not all input sizes agree')
        
        means, quats, scales, opacities, colors = utils.send_all_to_device(
            [
                means,
                quats,
                scales,
                opacities,
                colors
            ]
        )

        new_inds = torch.arange(self.means.shape[0], self.means.shape[0]+N)
        
        if self.added_ind is None:
            self.added_ind = new_inds
        else:
            self.added_ind = torch.cat((self.added_ind, new_inds))

        self.means = torch.cat((self.means, means))
        self.quats = torch.cat((self.quats, quats))
        self.scales = torch.cat((self.scales, scales))
        self.opacities = torch.cat((self.opacities, opacities))
        self.colors = torch.cat((self.colors, colors))

        return new_inds