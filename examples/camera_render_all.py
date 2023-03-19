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
import os
import json
import pickle
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ray_tracing", help="whether to use ray tracing", action="store_true", default=False)
    args = parser.parse_args()

    # Set ray_tracing to the value supplied by the user (default is False)
    ray_tracing = args.ray_tracing

    with open('/home/zubairirshad/SAPIEN/camera_rig_sapien.pickle', 'rb') as f:
        loaded_dict = pickle.load(f)

    all_mat44 = loaded_dict['all_mat44']

    token = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Im1pcnNoYWQ3QGdhdGVjaC5lZHUiLCJpcCI6IjE3Mi4yMC4wLjEiLCJwcml2aWxlZ2UiOjEsImZpbGVPbmx5Ijp0cnVlLCJpYXQiOjE2NzkwNTc1NjMsImV4cCI6MTcxMDU5MzU2M30.EINKQG0OukgeDFNDJr3v8HnXzrKvhk7RnAcjwi1C_zo'

    # ray_tracing = True
    if ray_tracing:
        sapien.render_config.camera_shader_dir = "rt"
        sapien.render_config.viewer_shader_dir = "rt"
        sapien.render_config.rt_samples_per_pixel = 256  # change to 256 for less noise
        sapien.render_config.rt_use_denoiser = True  # change to True for OptiX denoiser

    # single-handle refrigerator#10797, 10905, 10849, 10373, 11260, 12054, 12249, 12252
    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)

    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)

    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([1, 2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([1, -2, 2], [1, 1, 1], shadow=True)
    scene.add_point_light([-1, 0, 1], [1, 1, 1], shadow=True)

    ids = ['10797', '10905', '10849', '10373', '11260', '12054', '12249', '12252']
    id = 10373

    for id in ids:
        id = 10373
        urdf_path = SAPIEN.asset.download_partnet_mobility(id, token)
        loader = scene.create_urdf_loader()
        loader.fix_root_link = True
        # urdf_path = '../assets/179/mobility.urdf'
        # load as a kinematic articulation
        asset = loader.load_kinematic(urdf_path)

        art = get_articulation(scene, 'KinematicArticulation')

        joint_angles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

        for joint_num, joint_angle in enumerate(joint_angles):
            # joint_angle =  0
            # joints['joint_1'].set_drive_property(stiffness=100.0, damping=0.0) 
            art.set_qpos(np.deg2rad(joint_angle))

            #for 2 joints
            # art.set_qpos([np.deg2rad(position), np.deg2rad(position1)])

            # print("joints", joints)
            assert asset, 'URDF not loaded.'

            # ---------------------------------------------------------------------------- #
            # Camera
            # ---------------------------------------------------------------------------- #
            near, far = 0.1, 100
            width, height = 320, 240
            camera = scene.add_camera(
                name="camera",
                width=width,
                height=height,
                fovy=np.deg2rad(35),
                near=near,
                far=far,
            )

            camera.set_pose(sapien.Pose(p=[0, 0, 0]))

            print('Intrinsic matrix\n', camera.get_intrinsic_matrix())

            camera_mount_actor = scene.create_actor_builder().build_kinematic()
            camera.set_parent(parent=camera_mount_actor, keep_pose=False)
            render_path = '/home/zubairirshad/SAPIEN/renders'

            render_dir = os.path.join(render_path, str(id))
            os.makedirs(render_dir, exist_ok=True)

            angle = str(joint_angle)+'_degree'

            render_dir = os.path.join(render_dir, angle)
            os.makedirs(render_dir, exist_ok=True)
            all_c2w = {}
            all_c2w['camera_angle_x'] = np.deg2rad(35)
            all_c2w['frames'] = []
            for i, mat44 in enumerate(all_mat44):
                camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))
                scene.step()  # make everything set
                scene.update_render()
                camera.take_picture()

                if ray_tracing:
                    rgba = camera.get_float_texture('Color')  # [H, W, 4]
                    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
                    rgba_pil = Image.fromarray(rgba_img)

                    rgb_save_dir = os.path.join(render_dir, 'rgb')
                    os.makedirs(rgb_save_dir, exist_ok=True)
                    rgb_name = 'r_'+str(i)
                    rgb_save_path = os.path.join(rgb_save_dir, rgb_name+'.png')
                    rgba_pil.save(rgb_save_path)

                    position = camera.get_float_texture('Position')  # [H, W, 4]

                    c2w = camera.get_model_matrix()
                    transforms_dict = {}
                    transforms_dict[rgb_name] = c2w.tolist()
                    all_c2w['frames'].append(transforms_dict)

                    # Depth
                    depth = -position[..., 2]
                    depth_image = (depth * 1000.0).astype(np.uint16)
                    depth_pil = Image.fromarray(depth_image)

                    depth_save_dir = os.path.join(render_dir, 'depth')
                    os.makedirs(depth_save_dir, exist_ok=True)
                    depth_save_path = os.path.join(depth_save_dir, 'depth'+ str(i)+'.png')

                    depth_pil.save(depth_save_path)

                else:
                    seg_labels = camera.get_uint32_texture('Segmentation')  # [H, W, 4]
                    colormap = sorted(set(ImageColor.colormap.values()))
                    color_palette = np.array([ImageColor.getrgb(color) for color in colormap],
                                            dtype=np.uint8)
                    # label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
                    label1_image = seg_labels[..., 1].astype(np.uint8)  # actor-level
    #                label0_pil = Image.fromarray(color_palette[label0_image])
                    seg_save_dir = os.path.join(render_dir, 'seg')
                    os.makedirs(seg_save_dir, exist_ok=True)
                    seg_save_path = os.path.join(seg_save_dir, 'm_'+str(i)+'.png')

                    # label0_pil.save(seg_save_path)
                    label1_pil = Image.fromarray(color_palette[label1_image])
                    label1_pil.save(seg_save_path)
                # print("done with image", i)

            print("Done with joint num", joint_num)

            json_save_path = os.path.join(render_dir, 'transforms.json')
            with open(json_save_path, "w") as fp:
                json.dump(all_c2w,fp)

        print("Done with ID", id)
        print("======================================\n\n") 

        scene.remove_node(asset)


if __name__ == '__main__':
    main()
