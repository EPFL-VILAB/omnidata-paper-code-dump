from torch import nn
import torch
from pytorch3d.renderer import BlendParams, hard_rgb_blend
from pytorch3d.renderer.mesh.shading import flat_shading

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings,
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    HardFlatShader,
    TexturesUV,
    TexturesVertex,
)


class HardVeryFlatShader(nn.Module):
    """
    Per face lighting - the lighting model is applied using the average face
    position and the face normal. The blending function hard assigns
    the color of the closest face for each pixel.
    To use the default values, simply initialize the shader with the desired
    device e.g.
    .. code-block::
        shader = HardFlatShader(device=torch.device("cuda:0"))
    """

    def __init__(
        self, device="cpu", cameras=None, lights=None, materials=None, blend_params=None
    ):
        super().__init__()
        self.lights = lights if lights is not None else PointLights(device=device)
        self.materials = (
            materials if materials is not None else Materials(device=device)
        )
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def to(self, device):
        # Manually move to device modules which are not subclasses of nn.Module
        self.cameras = self.cameras.to(device)
        self.materials = self.materials.to(device)
        self.lights = self.lights.to(device)

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of HardFlatShader"
            raise ValueError(msg)
#         texels = meshes.sample_textures(fragments)
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
#         texels = torch.zeros_like(lights)
        texels = 0.5 * torch.tensor([1.0, 1.0, 1.0]).to(materials.device)
        blend_params = kwargs.get("blend_params", self.blend_params)
        colors = flat_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )
        images = hard_rgb_blend(colors, fragments, blend_params)
        return images, fragments

