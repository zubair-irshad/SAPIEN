import numpy as np
import open3d as o3d
import sapien.core as sapien

def convert_pose(C2W):
    flip_yz = np.eye(4)
    flip_yz[1, 1] = -1
    flip_yz[2, 2] = -1
    C2W = np.matmul(C2W, flip_yz)
    return C2W

def mat44_to_model_matrix(M):
    T = np.array(([0,0,-1],
                 [-1,0,0],
                 [0,1,0]))
    c2w = T @ M
    return c2w

def opengl_to_sapien(M):
    T = np.array(([0,-1,0],
                 [0,0,1],
                 [-1,0,0]))
    c2w = T @ M
    return c2w

def normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-9)

def get_all_c2w(all_cam_positions):

    # T = np.array(([0,-1,0,0],
    #               [1,0,0,0],
    #               [0,0,1,0],
    #               [0,0,0,1]))
    all_c2w = []
    for cam_pos in all_cam_positions:
        forward = -cam_pos / np.linalg.norm(cam_pos)
        left = np.cross([0, 0, 1], forward)
        left = left / np.linalg.norm(left)
        up = np.cross(forward, left)
        mat44 = np.eye(+4)
        mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        mat44[:3, 3] = cam_pos
        all_c2w.append(mat44)
        # forward = -cam_pos / np.linalg.norm(cam_pos)
        # left = np.cross([0, 0, 1], forward)
        # left = left / np.linalg.norm(left)
        # up = np.cross(forward, left)
        # mat44 = np.eye(4)
        # mat44[:3, :3] = np.stack([forward, left, up], axis=1)
        # # mat = convert_pose(look_at(cam_pos)[0])
        # mat44[:3, 3] = cam_pos
        # # mat = convert_pose(look_at(cam_pos)[0])
        # # mat[:3,:3] = opengl_to_sapien(mat[:3,:3])
        # all_c2w.append(mat44)
    print("len(all_c2w)", len(all_c2w), all_c2w[0].shape)
    return all_c2w

def look_at(cam_location):
    # Cam points in positive z direction
    forward = - cam_location
    forward = normalize(forward)

    # tmp = np.array([0., -1., 0.])
    tmp = np.array([0., -1., 0.])

    right = np.cross(tmp, forward)
    right = normalize(right)

    up = np.cross(forward, right)
    up = normalize(up)

    mat = np.stack((right, up, forward, cam_location), axis=-1)

    hom_vec = np.array([[0., 0., 0., 1.]])

    if len(mat.shape) > 2:
        hom_vec = np.tile(hom_vec, [mat.shape[0], 1, 1])

    mat = np.concatenate((mat, hom_vec), axis=-2)
    return mat, forward

def sample_spherical(n, radius=4):

    # n1 = int(0.7 * n)
    # x = np.random.uniform(-1, 1, size=(n1,1))
    # y = np.random.uniform(-1, 0, size=(n1,1))
    # z = np.random.uniform(-1, 1, size=(n1,1))
    # xyz1 = np.column_stack((x, y, z))
    # print("xyz1", xyz1.shape)
    # xyz1 = normalize(xyz1) * radius


    # n2 = int(0.3 * n)
    # x = np.random.uniform(-1, 1, size=(n2,1))
    # y = np.random.uniform(0, 1, size=(n2,1))
    # z = np.random.uniform(-1, 1, size=(n2,1))
    # xyz2 = np.column_stack((x, y, z))
    # # xyz2 = np.random.uniform(-1, 1, size=(n2,3))
    # xyz2 = normalize(xyz2) * radius

    # print("xyz1", xyz1.shape, xyz2.shape)
    # xyz = np.vstack((xyz1, xyz2))
    xyz = np.random.normal(size=(n,3))
    xyz[:,2] = np.absolute(xyz[:,2])
    xyz = normalize(xyz) * radius
    return xyz

def get_articulation(scene, name) -> sapien.ArticulationBase:
    all_articulations = scene.get_all_articulations()
    print("all_articulations", all_articulations)
    return all_articulations[0]

def get_joints_dict(articulation: sapien.Articulation):
    joints = articulation.get_joints()
    joint_names =  [joint.name for joint in joints]
    assert len(joint_names) == len(set(joint_names)), 'Joint names are assumed to be unique.'
    return {joint.name: joint for joint in joints}

def get_camera_frustum(img_size, focal, C2W, frustum_length=0.5, color=[0., 1., 0.]):
    W, H = img_size
    hfov = np.rad2deg(np.arctan(W / 2. / focal) * 2.)
    vfov = np.rad2deg(np.arctan(H / 2. / focal) * 2.)
    half_w = frustum_length * np.tan(np.deg2rad(hfov / 2.))
    half_h = frustum_length * np.tan(np.deg2rad(vfov / 2.))

    # build view frustum for camera (I, 0)
    frustum_points = np.array([[0., 0., 0.],                          # frustum origin
                               [-half_w, -half_h, frustum_length],    # top-left image corner
                               [half_w, -half_h, frustum_length],     # top-right image corner
                               [half_w, half_h, frustum_length],      # bottom-right image corner
                               [-half_w, half_h, frustum_length]])    # bottom-left image corner
    frustum_lines = np.array([[0, i] for i in range(1, 5)] + [[i, (i+1)] for i in range(1, 4)] + [[4, 1]])
    frustum_colors = np.tile(np.array(color).reshape((1, 3)), (frustum_lines.shape[0], 1))

    # frustum_colors = np.vstack((np.tile(np.array([[1., 0., 0.]]), (4, 1)),
    #                            np.tile(np.array([[0., 1., 0.]]), (4, 1))))

    # transform view frustum from (I, 0) to (R, t)
    # C2W = np.linalg.inv(W2C)
    frustum_points = np.dot(np.hstack((frustum_points, np.ones_like(frustum_points[:, 0:1]))), C2W.T)
    frustum_points = frustum_points[:, :3] / frustum_points[:, 3:4]

    return frustum_points, frustum_lines, frustum_colors

def frustums2lineset(frustums):
    N = len(frustums)
    merged_points = np.zeros((N*5, 3))      # 5 vertices per frustum
    merged_lines = np.zeros((N*8, 2))       # 8 lines per frustum
    merged_colors = np.zeros((N*8, 3))      # each line gets a color

    for i, (frustum_points, frustum_lines, frustum_colors) in enumerate(frustums):
        merged_points[i*5:(i+1)*5, :] = frustum_points
        merged_lines[i*8:(i+1)*8, :] = frustum_lines + i*5
        merged_colors[i*8:(i+1)*8, :] = frustum_colors

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(merged_points)
    lineset.lines = o3d.utility.Vector2iVector(merged_lines)
    lineset.colors = o3d.utility.Vector3dVector(merged_colors)

    return lineset