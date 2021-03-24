import re
from collections import namedtuple
import torch
import transforms3d.euler as euler
import transforms3d.axangles as axangles
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import FoVPerspectiveCameras

LabelFile = namedtuple('LabelFile', ['point', 'view', 'domain'])
View = namedtuple('View', ['point', 'view'])

def parse_filename( filename ):
    p = re.match('.*point_(?P<point>\d+)_view_(?P<view>\d+)_domain_(?P<domain>\w+)', filename)
    if p is None:
        raise ValueError( 'Filename "{}" not matched. Must be of form point_XX_view_YY_domain_ZZ.**.'.format(filename) )
    return {'point': p.group('point'), 'view': p.group('view'), 'domain': p.group('domain') }



def load_taskonomy_obj_to_mesh(obj_filename, device='cpu'):
    '''
        Loads taskonomy mesh. On disk, it appears to be stored under some rotation.
        This applies another rotation that kind of undoes what's on disk. 
        We still have to apply some transforms on top of the taskonomy information.
        Basically:
        mesh = H1 * taskonomy_mesh
        camera = H2 * taskonomy_camera
    '''
    verts, faces, aux = load_obj(obj_filename, device=device)
    H = torch.tensor([
            [-1, 0, 0],
            [0, 0, -1],
            [ 0, -1, 0],
    ], dtype=torch.float32).to(verts.device)
    verts = (H.matmul(verts.T)).T
    return Meshes(verts=[verts], faces=[faces.verts_idx])

def get_RT_from_taskonomy_camera(location, rotation):
    '''
       
    '''
    Tx, Ty, Tz = location
    ex, ey, ez = rotation
        
    (x,y,z), w = euler.euler2axangle(ex, ey, ez, 'sxyz')
    R = torch.tensor(axangles.axangle2mat(torch.tensor([x, -y, z]), w)).unsqueeze(0)
    T = torch.tensor([[-Tx, Ty, -Tz]], dtype=torch.float64)
    return R, T

def create_taskonomy_camera(point_info, device='cpu'):
    location = point_info['camera_location']
    rotation = point_info['camera_rotation_final']
    fov = point_info['field_of_view_rads']
    R, T = get_RT_from_taskonomy_camera(location, rotation)
    T_inv = -R.bmm(T.unsqueeze(-1)).squeeze(-1)
    R_inv = R.transpose(1, 2)
#     return (R_inv, T_inv, fov)
    return FoVPerspectiveCameras(device=device, R=R_inv, T=T_inv, fov=fov, degrees=False)