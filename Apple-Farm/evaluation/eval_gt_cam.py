import pix4d_io as pix4d
from pathlib import Path

from subprocess import call, run

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import shutil


def read_used_names(PATH_MODEL):
    """
    Read images that where used for sparse model
    Returns:  ref_names(list of image names)
    """
    TXT_PATH = f"{PATH_MODEL}"
    # Read image names from images.txt that where used for sparse model
    images_df = pd.read_csv(f"{TXT_PATH}/images.txt", sep=" ", comment="#", header=None, usecols=range(10))
    images_df.columns = ["IMAGE_ID", "QW", "QX", "QY", "QZ", "TX", "TY", "TZ", "CAMERA_ID", "NAME"]

    images_df = images_df.drop(images_df.index[1::2])
    ref_names = images_df["NAME"].tolist()

    return ref_names


# Function to read image names from a file into a list
def read_image_list(image_list_path):
    with open(image_list_path, "r") as file:
        return [line.strip() for line in file if line.strip()]


# Function to read offset values from a file into a tuple
def read_offset(offset_path):
    with open(offset_path, "r") as file:
        offset = file.read().strip().split()
        return tuple(float(value) for value in offset)


# -------------------------(offset & image list) gt camera poses ---------------------------------

# 2cross seasons
PROJECT_PATH = Path("/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case2/project_seasons")
OUTPUT_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/gt_camera_pose")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
for i in range(2):
    if i == 0:  # winter
        winter_file = f"/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case2/winter"
        file_extern_cam_file = f"{winter_file}/applefarm-230209_merge_v9_calibrated_external_camera_parameters.txt"
        IMAGE_LIST_PATH = f"{winter_file}/image_list_winter.txt"
        # OFFSET_PATH = '2_different_season/winter/applefarm-230209_merge_v9_offset.xyz'
        output_filename = OUTPUT_PATH / f"gt_camera_poses_m{i}.txt"  #'gt_w_camera_poses.txt'
    elif i == 1:  # summer
        summer_file = "/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case2/summer"
        file_extern_cam_file = f"{summer_file}/apple-farm-20230627-north_calibrated_external_camera_parameters.txt"
        IMAGE_LIST_PATH = f"{summer_file}/image_list_summer.txt"
        # OFFSET_PATH = '2_different_season/summer/apple-farm-20230627-north_offset.xyz'
        # OFFSET_PATH = '2_different_season/winter/applefarm-230209_merge_v9_offset.xyz'
        output_filename = OUTPUT_PATH / f"gt_camera_poses_m{i}.txt"  #'gt_sn_camera_poses.txt'


# diff_views
PROJECT_PATH = Path("/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case1/project_views")
OUTPUT_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/gt_camera_pose")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
for i in range(3):
    if i == 0:  # winter
        winter_file = f"/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case1/winter"
        file_extern_cam_file = f"{winter_file}/applefarm-230209_merge_v9_calibrated_external_camera_parameters.txt"
        IMAGE_LIST_PATH = f"{winter_file}/image_list_winter.txt"
        output_filename = OUTPUT_PATH / f"gt_camera_poses_m{i}.txt"  #'gt_w_camera_poses.txt'
    elif i == 1:  # summer_east
        file_extern_cam_file = f"/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case1/summer_east/apple-farm-230209-east_calibrated_external_camera_parameters.txt"
        IMAGE_LIST_PATH = f"/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case1/summer_east/image_list_east.txt"
        output_filename = OUTPUT_PATH / f"gt_camera_poses_m{i}.txt"  #'gt_se_camera_poses.txt'
    elif i == 2:  # summer_north
        file_extern_cam_file = f"/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case1/summer_north/apple-farm-20230627-north_calibrated_external_camera_parameters.txt"
        IMAGE_LIST_PATH = f"/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case1/summer_north/image_list_north.txt"
        output_filename = OUTPUT_PATH / f"gt_camera_poses_m{i}.txt"  #'gt_sn_camera_poses.txt'


# 4 andreas park
PROJECT_PATH = Path("/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/project_new_andreas_zigzagpair")
OUTPUT_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/gt_camera_pose")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
for i in range(3):
    if i == 0:  # 30zigzag
        IMAGE_LIST_PATH = (
            f"/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/dataset_andreas_park/image_list_30zigzag.txt"
        )
        output_filename = OUTPUT_PATH / f"gt_camera_poses_m{i}.txt"  #'gt_sn_camera_poses.txt'
        file_extern_cam_file = f"/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/dataset_andreas_park/andreas-park-20240130-params/params/andreas-park-20240130-all_calibrated_external_camera_parameters.txt"
    elif i == 1:
        IMAGE_LIST_PATH = (
            f"/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/dataset_andreas_park/image_list_27zigzag_1.txt"
        )
        output_filename = OUTPUT_PATH / f"gt_camera_poses_m{i}.txt"  #'gt_w_camera_poses.txt'
        file_extern_cam_file = f"/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/dataset_andreas_park/andreas-park-20240127-params/andreas-park-20240127-all_calibrated_external_camera_parameters.txt"
    elif i == 2:
        IMAGE_LIST_PATH = (
            f"/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/dataset_andreas_park/image_list_27zigzag_2.txt"
        )
        output_filename = OUTPUT_PATH / f"gt_camera_poses_m{i}.txt"  #'gt_w_camera_poses.txt'
        file_extern_cam_file = f"/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/dataset_andreas_park/andreas-park-20240127-params/andreas-park-20240127-all_calibrated_external_camera_parameters.txt"

    # # 4 andreas park_entire
    # PROJECT_PATH = Path('/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/project_andreas_zigzagentire')

    # OUTPUT_PATH = Path(f'{PROJECT_PATH}/Correctness_loop/evaluation/gt_camera_pose')
    # OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    # for i in range(2):

    #     if i == 1: # 30zigzag
    #         IMAGE_LIST_PATH = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/dataset_andreas_park/image_list_30zigzag.txt'
    #         output_filename = OUTPUT_PATH / f'gt_camera_poses_m{i}.txt' #'gt_sn_camera_poses.txt'
    #         file_extern_cam_file = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/dataset_andreas_park/andreas-park-20240130-params/params/andreas-park-20240130-all_calibrated_external_camera_parameters.txt'
    #     elif i == 0:
    #         IMAGE_LIST_PATH = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/dataset_andreas_park/image_list_27zigzag.txt'
    #         output_filename = OUTPUT_PATH / f'gt_camera_poses_m{i}.txt' #'gt_w_camera_poses.txt'
    #         file_extern_cam_file = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/dataset_andreas_park/andreas-park-20240127-params/andreas-park-20240127-all_calibrated_external_camera_parameters.txt'

    image_names = read_image_list(IMAGE_LIST_PATH)
    # INIT_ALIGNED_MODEL_PATH = f'{PROJECT_PATH}/Correctness_loop/init_ICP_swisstopo/Output/Init_transed_models'
    INIT_MODEL_PATH = f"{PROJECT_PATH}/Models/Model_{i}/shifted_LV95_model_txt"
    ref_names = read_used_names(INIT_MODEL_PATH)
    # offset = read_offset(OFFSET_PATH)
    # apple farm
    cx = 2678199.0
    cy = 1258409.0
    cz = 445.000
    # # andreas 2684399.0 1251969.0 200.0
    # cx = 2684399.0
    # cy = 1251969.0
    # cz = 200.0
    offset = (cx, cy, cz)
    # print('offset',offset)

    points, orientations, filenames = pix4d.load_pix4d_sfm_model(file_extern_cam_file, ref_names, offset)

    with open(output_filename, "w") as f:
        f.write("filename, x, y, z, qw, qx, qy, qz\n")
        for filename, point, orientation in zip(filenames, points, orientations):
            f.write(
                f"{filename}, {point[0]}, {point[1]}, {point[2]}, {orientation[0]}, {orientation[1]}, {orientation[2]}, {orientation[3]}\n"
            )
    print(f"model{i} done!")

    CAM_COMPARE_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/Camera_poses_compare")
    CAM_COMPARE_PATH.mkdir(parents=True, exist_ok=True)
    cam_compare_file_gt = f"{CAM_COMPARE_PATH}/cam_gt_m{i}.txt"
    with open(cam_compare_file_gt, "w") as f:
        f.write("filename qw qx qy qz x y z\n")
        for filename, point, orientation in zip(filenames, points, orientations):
            f.write(
                f"{filename} {orientation[0]} {orientation[1]} {orientation[2]} {orientation[3]} {point[0]} {point[1]} {point[2]}\n"
            )
