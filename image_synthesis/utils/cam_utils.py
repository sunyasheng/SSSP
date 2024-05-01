import torch
import numpy as np
from .pytorch3d_transforms import matrix_to_euler_angles, euler_angles_to_matrix


def compute_rotation(angles):
    """
    Return:
        rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat
    Parameters:
        angles           -- torch.tensor, size (B, 3), radian
    """

    batch_size = angles.shape[0]
    ones = torch.ones([batch_size, 1]).to(angles)
    zeros = torch.zeros([batch_size, 1]).to(angles)
    x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
    
    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x), 
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([batch_size, 3, 3])
    
    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([batch_size, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)



def compute_angles(rotations):
    batch_size = rotations.shape[0]
    
    sy = torch.sqrt(rotations[:,0,0]**2+rotations[:,1,0]**2)
    # import pdb; pdb.set_trace()
    assert torch.all(sy>1e-6)
    angle_x = torch.atan2(rotations[:,2,1], rotations[:,2,2])
    angle_y = torch.atan2(-rotations[:,2,0], sy)
    angle_z = torch.atan2(rotations[:,1,0], rotations[:,0,0])
    angle = torch.stack([angle_x,angle_y,angle_z],dim=1)
    return angle



######## for translating to camera params conforming to eg3d protocal
def angle_translation_to_camera(angle_translation):
    batch_size = angle_translation.shape[0]
    intrinsics = np.array([0.0, 0.0, 0.0, 1.0, 4.2647, 0.0, 0.5, 0.0, 4.2647, 0.5, 0.0, 0.0, 1.0])
    intrinsics = torch.from_numpy(intrinsics).to(angle_translation)
    intrinsics = intrinsics.unsqueeze(0).repeat(batch_size, 1)

    angle = angle_translation[:, :3]
    translation = angle_translation[:, 3:].unsqueeze(-1)
    rotation = euler_angles_to_matrix(angle, 'ZYX')
    # import pdb; pdb.set_trace();
    rotation_translation_cat = torch.cat([rotation, translation], dim=2)

    cat = [rotation_translation_cat.reshape(batch_size,12), intrinsics]
    camera_params = torch.cat(cat, dim=1)

    return camera_params


if __name__ == '__main__':
    cam = torch.from_numpy(np.array([[[0.9995401501655579, 0.011665810830891132, -0.027987297624349594, 0.069649438753417, 
                                        0.006274465937167406, -0.9826249480247498, -0.18549618124961853, 0.46442171799954096, 
                                        -0.029664980247616768, 0.18523527681827545, -0.9822463393211365, 2.6588458818689906]]]))
    cam = cam.reshape(1, 3, 4)
    # angles = compute_angles(cam[:,:3,:3])
    euler_angles = matrix_to_euler_angles(cam[:,:3,:3], 'ZYX')
    cycle_rot = euler_angles_to_matrix(euler_angles, 'ZYX')

    trans = cam[:, :3, 3]
    cam_params = angle_translation_to_camera(torch.cat([euler_angles, trans], dim=1))
    import pdb; pdb.set_trace();
