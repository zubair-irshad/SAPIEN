import sys
sys.path.append('/home/zubairirshad/SAPIEN')
from examples.rendering.utils import *

all_cam_positions = sample_spherical(60)
print("all_cam_positions", all_cam_positions[0])
all_mat44 = get_all_c2w(all_cam_positions)

camera_rig_dict = {}
camera_rig_dict['all_mat44'] = all_mat44

import pickle
with open('camera_rig_sapien.pickle', 'wb') as handle:
    pickle.dump(camera_rig_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)