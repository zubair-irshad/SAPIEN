"""Camera.

Concepts:
    - Create and mount cameras
    - Render RGB images, point clouds, segmentation masks
"""

import sapien.core as sapien
import numpy as np
from PIL import Image, ImageColor
import open3d as o3d
from sapien.utils.viewer import Viewer
from transforms3d.euler import mat2euler
import sapien as SAPIEN
import sys
sys.path.append('/home/zubairirshad/SAPIEN')
from examples.rendering.utils import *
import math
import pytransform3d.visualizer as pv
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray_tracing", help="whether to use ray tracing", action="store_true", default=False)
    args = parser.parse_args()

    # Set ray_tracing to the value supplied by the user (default is False)
    ray_tracing = args.ray_tracing

    fig = pv.figure()
    things_to_draw = []
    all_cam_positions = sample_spherical(60)
    print("len(all cam pos", len(all_cam_positions))
    print("all_cam_positions", all_cam_positions[0])
    all_c2w = get_all_c2w(all_cam_positions)
    token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1pcnNoYWQ3QGdhdGVjaC5lZHUiLCJpcCI6IjE3Mi4yMC4wLjEiLCJwcml2aWxlZ2UiOjEsImZpbGVPbmx5Ijp0cnVlLCJpYXQiOjE2NzkwNTc1NjMsImV4cCI6MTcxMDU5MzU2M30.EINKQG0OukgeDFNDJr3v8HnXzrKvhk7RnAcjwi1C_zo'

    # single-handle refrigerator#10797, 10905, 10849, 10373, 11260, 12054, 12249, 12252
    urdf_path = SAPIEN.asset.download_partnet_mobility(10373, token)
    # create scene and URDF loader
    # urdf_loader.load(urdf_file)
    ray_tracing = True
    if ray_tracing:
        sapien.render_config.camera_shader_dir = "rt"
        sapien.render_config.viewer_shader_dir = "rt"
        sapien.render_config.rt_samples_per_pixel = 256  # change to 256 for less noise
        sapien.render_config.rt_use_denoiser = True  # change to True for OptiX denoiser

    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    # urdf_path = '../assets/179/mobility.urdf'
    # load as a kinematic articulation
    asset = loader.load_kinematic(urdf_path)

    art = get_articulation(scene, 'KinematicArticulation')
    # joints = asset.get_joints()
    # joints = get_joints_dict(asset)
    # print(joints.keys())
    # for i in range()
    position =  0
    # joints['joint_1'].set_drive_property(stiffness=100.0, damping=0.0) 
    art.set_qpos(np.deg2rad(position))

    #for 2 joints
    # art.set_qpos([np.deg2rad(position), np.deg2rad(position1)])

    # print("joints", joints)
    assert asset, 'URDF not loaded.'


    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

    # ---------------------------------------------------------------------------- #
    # Camera
    # ---------------------------------------------------------------------------- #
    near, far = 0.1, 100
    width, height = 640, 480
    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(35),
        near=near,
        far=far,
    )
    # camera.set_pose(sapien.Pose(p=[1, 0, 0]))

    camera.set_pose(sapien.Pose(p=[0, 0, 0]))

    print('Intrinsic matrix\n', camera.get_intrinsic_matrix())

    camera_mount_actor = scene.create_actor_builder().build_kinematic()
    camera.set_parent(parent=camera_mount_actor, keep_pose=False)

#    Compute the camera pose by specifying forward(x), left(y) and up(z)
    # cam_pos = np.array([-2, 2, 3])
    # forward = -cam_pos / np.linalg.norm(cam_pos)
    # left = np.cross([0, 0, 1], forward)
    # left = left / np.linalg.norm(left)
    # up = np.cross(forward, left)
    # mat44 = np.eye(+4)
    # mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    # mat44[:3, 3] = cam_pos
    
    mat44 = all_c2w[0]
    camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))


    all_c2w_new = []
    for mat44_new in all_c2w:
        mat44_new[:3,:3] = mat44_to_model_matrix(mat44[:3,:3])
        all_c2w_new.append(mat44_new)

    scene.step()  # make everything set
    scene.update_render()
    camera.take_picture()

    # ---------------------------------------------------------------------------- #
    # RGBA
    # ---------------------------------------------------------------------------- #
    rgba = camera.get_float_texture('Color')  # [H, W, 4]
    # An alias is also provided
    # rgba = camera.get_color_rgba()  # [H, W, 4]
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    rgba_pil = Image.fromarray(rgba_img)
    rgba_pil.save('color.png')

    # ---------------------------------------------------------------------------- #
    # XYZ position in the camera space
    # ---------------------------------------------------------------------------- #
    # Each pixel is (x, y, z, render_depth) in camera space (OpenGL/Blender)
    position = camera.get_float_texture('Position')  # [H, W, 4]

    # OpenGL/Blender: y up and -z forward
    points_opengl = position[..., :3][position[..., 3] < 1]
    points_color = rgba[position[..., 3] < 1][..., :3]
    # Model matrix is the transformation from OpenGL camera space to SAPIEN world space
    # camera.get_model_matrix() must be called after scene.update_render()!
    model_matrix = camera.get_model_matrix()
    points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]



    
    # SAPIEN CAMERA: z up and x forward
    # points_camera = points_opengl[..., [2, 0, 1]] * [-1, -1, 1]

    c2w = camera.get_model_matrix()

    c2w_3by3 = c2w[:3,:3]
    mat44_3by3 = mat44[:3,:3]

    T = np.linalg.inv(mat44_3by3) @ c2w_3by3

    print("T", T)
    # c2w_new = T @ mat44[:3,:3]

    # print("c2w_new", c2w_new[:3,:3])
    # print("c2w", c2w)
    # print("c2w", c2w[:3,:3])
    # print("mat44", mat44[:3,:3])
    # print("mat44", np.linalg.inv(mat44[:3,:3]))


    # print("mat44", mat44)

    intrinsics = camera.get_intrinsic_matrix()
    frustums = []
    for C2W in all_c2w_new:
        frustums.append(get_camera_frustum((width, height), intrinsics[0,0], convert_pose(C2W), frustum_length=0.2, color=[0,1,0]))
    cameras = frustums2lineset(frustums)
    things_to_draw.append(cameras)

    for C2W in all_c2w_new:
        fig.plot_transform(A2B=convert_pose(C2W), s=0.2, strict_check=False)
    # intrinsics = camera.get_intrinsic_matrix()
    frustums = []
    frustums.append(get_camera_frustum((width, height), intrinsics[0,0], convert_pose(c2w), frustum_length=2.0, color=(1,0,0)))
    cameras = frustums2lineset(frustums)
    things_to_draw.append(cameras)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_world)
    pcd.colors = o3d.utility.Vector3dVector(points_color)

    max_bounds = pcd.get_max_bound()
    min_bounds = pcd.get_min_bound()
    print("max_bounds", max_bounds)
    print("min_bounds", min_bounds)

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))

    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()

    obj_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    obj_coord_frame = obj_coord_frame.transform(convert_pose(c2w))
    
    things_to_draw.append(pcd)
    things_to_draw.append(coord_frame)
    things_to_draw.append(sphere)
    things_to_draw.append(cameras)
    things_to_draw.append(obj_coord_frame)
    for geometry in things_to_draw:
        fig.add_geometry(geometry)

    #fig.add_geometry([pcd, coord_frame, sphere, cameras, obj_coord_frame])
    fig.show()
    #o3d.visualization.draw_geometries([pcd, coord_frame, sphere, cameras, obj_coord_frame])

    # Depth
    depth = -position[..., 2]
    depth_image = (depth * 1000.0).astype(np.uint16)
    depth_pil = Image.fromarray(depth_image)
    depth_pil.save('depth.png')

    # ---------------------------------------------------------------------------- #
    # Segmentation labels
    # ---------------------------------------------------------------------------- #
    # Each pixel is (visual_id, actor_id/link_id, 0, 0)
    # visual_id is the unique id of each visual shape
    seg_labels = camera.get_uint32_texture('Segmentation')  # [H, W, 4]
    colormap = sorted(set(ImageColor.colormap.values()))
    color_palette = np.array([ImageColor.getrgb(color) for color in colormap],
                             dtype=np.uint8)
    label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
    label1_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
    # Or you can use aliases below
    # label0_image = camera.get_visual_segmentation()
    # label1_image = camera.get_actor_segmentation()
    label0_pil = Image.fromarray(color_palette[label0_image])
    label0_pil.save('label0.png')
    label1_pil = Image.fromarray(color_palette[label1_image])
    label1_pil.save('label1.png')

    # ---------------------------------------------------------------------------- #
    # Take picture from the viewer
    # ---------------------------------------------------------------------------- #
    viewer = Viewer(renderer)
    viewer.set_scene(scene)
    # We show how to set the viewer according to the pose of a camera
    # opengl camera -> sapien world
    model_matrix = camera.get_model_matrix()

    print("model_matrix", model_matrix.shape)
    # sapien camera -> sapien world
    # You can also infer it from the camera pose
    model_matrix = model_matrix[:, [2, 0, 1, 3]] * np.array([-1, -1, 1, 1])
    # The rotation of the viewer camera is represented as [roll(x), pitch(-y), yaw(-z)]
    rpy = mat2euler(model_matrix[:3, :3]) * np.array([1, -1, -1])
    viewer.set_camera_xyz(*model_matrix[0:3, 3])
    viewer.set_camera_rpy(*rpy)
    viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
    while not viewer.closed:
        if viewer.window.key_down('p'):  # Press 'p' to take the screenshot
            rgba = viewer.window.get_float_texture('Color')
            rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
            rgba_pil = Image.fromarray(rgba_img)
            rgba_pil.save('screenshot.png')
        scene.step()
        scene.update_render()
        viewer.render()


if __name__ == '__main__':
    main()
