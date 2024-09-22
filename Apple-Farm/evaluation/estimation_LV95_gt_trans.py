# Model Aligner for sparse reconstruction
import argparse
import pandas as pd
from pathlib import Path
from subprocess import call, run
import shutil
import pycolmap
import numpy as np
from scipy.spatial.transform import Rotation as R
import pix4d_io as pix4d
import sys
import os

apple_farm_path = os.path.join(os.path.dirname(__file__), "..", "src")
if apple_farm_path not in sys.path:
    sys.path.insert(0, apple_farm_path)
from colmap_vidual import colmap_to_txt


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


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # -                           ARGPARSE                             -
    # ------------------------------------------------------------------

    # Argparse: Create parser
    parser = argparse.ArgumentParser()
    # Argparse: Add arguments
    parser.add_argument(
        dest="project_path", help="Path to the project folder, containing Models/Model_i folders with sparse models"
    )
    parser.add_argument(
        "--image_lists",
        nargs="+",
        help="Path to txt file which indicates which images should be used. List of image names",
    )
    parser.add_argument("--gt_cams", nargs="+", help=" .txt file which contains gt camera poses from pix4d")
    parser.add_argument("--offset", nargs=3, type=float, help="offset value [x,y,z] for the gt camera sequences")
    parser.add_argument(
        "--colmap_command",
        dest="COLMAP",
        default="colmap",
        help="Change if colmap is installed locally at specific path ./path/to/colmap",
    )
    # Argparse: Parse arguments
    args = parser.parse_args()

    COLMAP = args.COLMAP
    PROJECT_PATH = Path(f"{args.project_path}")
    MODEL_DIR = Path(f"{args.project_path}/Models")
    EVALUATION_DIR = Path(f"{args.project_path}/Correctness_loop/evaluation")

    IMAGES_LV95_BATCHES = f"{args.project_path}/Images_LV95_batches.csv"
    # Read number of batches/models
    df_batches = pd.read_csv(IMAGES_LV95_BATCHES)
    max_batch = df_batches["batch"].max()
    nr_batches = int(max_batch) + 1
    print("nr of batches:", nr_batches)

    # ------------------------------------------------------------------
    # -                                    GT                          -
    # ------------------------------------------------------------------
    OUTPUT_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/gt_camera_pose")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    for i in range(nr_batches):
        IMAGE_LIST_PATH = Path(args.image_lists[i])
        file_extern_cam_file = Path(args.gt_cams[i])
        output_filename = OUTPUT_PATH / f"gt_camera_poses_m{i}.txt"

        image_names = read_image_list(IMAGE_LIST_PATH)
        INIT_MODEL_PATH = f"{PROJECT_PATH}/Models/Model_{i}/shifted_LV95_model_txt"
        ref_names = read_used_names(INIT_MODEL_PATH)

        # offset = read_offset(OFFSET_PATH)
        cx = float(args.offset[0])
        cy = float(args.offset[1])
        cz = float(args.offset[2])
        offset = (cx, cy, cz)

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

    # ------------------------------------------------------------------
    # -           Transform (shifted) Colmap initial_models TO GT         -
    # ------------------------------------------------------------------
    for idx in range(nr_batches):
        print(f"Transforming model_{idx}...to GT")
        # Input model
        shifted_model_path = f"{MODEL_DIR}/Model_{idx}/shifted_LV95_model"
        # reference GT
        CAM_GT_PATH = Path(f"{EVALUATION_DIR}/gt_camera_pose/gt_camera_poses_m{idx}.txt")

        with CAM_GT_PATH.open("r") as file:
            lines = file.readlines()
        ref_cams_gt = []
        for line in lines[1:]:
            parts = line.strip().split(",")
            ref_cam_gt = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}\n"
            ref_cams_gt.append(ref_cam_gt)

        REF_IMAGE_PATH = Path(f"{EVALUATION_DIR}/gt_camera_pose/georeg_ref_image_{idx}.txt")
        with REF_IMAGE_PATH.open("w") as new_file:
            new_file.writelines(ref_cams_gt)

        # Model Aligner
        # -----------------------------------------------------------------------
        # Create output path to store GT (geo-registered colmap) model
        GEO_REG_PATH = Path(f"{EVALUATION_DIR}/gt_models/GT_Model_{idx}")
        GEO_REG_PATH.mkdir(parents=True, exist_ok=True)
        OUTPUT_TRANS_PATH = Path(f"{GEO_REG_PATH}/INIT_TRANS_GT_m{idx}.txt")

        # Align the model with the shifted LV95 coordinates of the images
        model_aligner_command = [
            f"{COLMAP}",
            "model_aligner",
            "--input_path",
            f"{shifted_model_path}",
            "--output_path",
            f"{GEO_REG_PATH}",
            "--transform_path",
            f"{OUTPUT_TRANS_PATH}",
            "--ref_images_path",
            f"{REF_IMAGE_PATH}",
            "--ref_is_gps",
            "0",  # cartesian-based
            "--alignment_type",
            "ecef",  # custom
            "--robust_alignment",
            "1",
            "--robust_alignment_max_error",
            "3.0",
        ]
        call(model_aligner_command)

        # Convert model to txt format
        GEO_TXT_PATH = Path(f"{EVALUATION_DIR}/gt_models/GT_Model_{idx}_txt")
        GEO_TXT_PATH.mkdir(parents=True, exist_ok=True)
        cam_pose_file = f"{GEO_TXT_PATH}/cam_gt_extract_m{idx}.txt"

        colmap_to_txt(GEO_REG_PATH, GEO_TXT_PATH, cam_pose_file)
        shutil.copy(
            f"{GEO_TXT_PATH}/cam_gt_extract_m{idx}.txt",
            f"{EVALUATION_DIR}/Camera_poses_compare/cam_gt_extract_m{idx}.txt",
        )

    # ------------------------------------------------------------------
    # -    other models follow the transformation of the base model    -
    # ------------------------------------------------------------------
    for idx in range(1, nr_batches):
        print(f"transforming model{idx} accordingly")

        shifted_model_path = f"{MODEL_DIR}/Model_{idx}/shifted_LV95_model"
        shifted_model = pycolmap.Reconstruction(shifted_model_path)

        gt_tran_file = Path(f"{EVALUATION_DIR}/gt_models/GT_Model_0/INIT_TRANS_GT_m0.txt")
        gt_tran = np.loadtxt(gt_tran_file)

        shifted_model.transform(pycolmap.SimilarityTransform3(gt_tran[:3]))
        OUTPUT_MODEL_PATH = Path(f"{EVALUATION_DIR}/gt_models/GTAlign_Init_Model_{idx}")
        OUTPUT_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        shifted_model.write(OUTPUT_MODEL_PATH)

        # VIDUALIZATION
        TXT_PATH = Path(f"{EVALUATION_DIR}/gt_models/GTAlign_Init_Model_{idx}_txt")
        TXT_PATH.mkdir(parents=True, exist_ok=True)
        cam_pose_file = f"{TXT_PATH}/cam_eval_m{idx}.txt"
        colmap_to_txt(OUTPUT_MODEL_PATH, TXT_PATH, cam_pose_file)

        shutil.copy(
            f"{TXT_PATH}/cam_eval_m{idx}.txt", f"{EVALUATION_DIR}/Camera_poses_compare/cam_colmap_init_m{idx}.txt"
        )

    print("Done!")
