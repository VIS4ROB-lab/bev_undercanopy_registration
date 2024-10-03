# Model Aligner for sparse reconstruction
import argparse
import pandas as pd
from pathlib import Path
from subprocess import call, run
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R
from colmap_vidual import colmap_to_txt


def read_used_names(PATH_MODEL, COLMAP):
    """
    Read images that where used for sparse model

    Returns:
    ref_names: list of image names
    """

    TXT_PATH = f"{PATH_MODEL}/model_txt"

    run(["mkdir", TXT_PATH], check=True)

    # Temporarily create conversion of sparse model to txt
    model_converter_command = [
        f"{COLMAP}",
        "model_converter",
        "--input_path",
        f"{PATH_MODEL}",
        "--output_path",
        f"{TXT_PATH}",
        "--output_type",
        "TXT",
    ]
    call(model_converter_command)

    # Read image names from images.txt that where used for sparse model
    images_df = pd.read_csv(f"{TXT_PATH}/images.txt", sep=" ", comment="#", header=None, usecols=range(10))
    images_df.columns = ["IMAGE_ID", "QW", "QX", "QY", "QZ", "TX", "TY", "TZ", "CAMERA_ID", "NAME"]

    images_df = images_df.drop(images_df.index[1::2])

    ref_names = images_df["NAME"].tolist()

    # Delete model_txt again
    shutil.rmtree(TXT_PATH)

    return ref_names


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
        "--colmap_command",
        dest="COLMAP",
        default="colmap",
        help="Change if colmap is installed locally at specific path ./path/to/colmap",
    )
    # Argparse: Parse arguments
    args = parser.parse_args()

    COLMAP = args.COLMAP
    MODEL_DIR = Path(f"{args.project_path}/Models")
    EVALUATION_DIR = Path(f"{args.project_path}/Correctness_loop/evaluation")
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # -                           Transform                            -
    # ------------------------------------------------------------------
    print("Transforming models to shifted LV95 coordinate system...")

    IMAGES_LV95_BATCHES = f"{args.project_path}/Images_LV95_batches.csv"

    # Read number of batches/models
    df_batches = pd.read_csv(IMAGES_LV95_BATCHES)
    max_batch = df_batches["batch"].max()
    nr_batches = int(max_batch) + 1
    print("nr of batches:", nr_batches)

    for idx in range(nr_batches):
        print(f"Transforming model_{idx}...")

        SPARSE_PATH = Path(f"{MODEL_DIR}/Model_{idx}/sparse/0")

        # Dataframe image data of current model
        df_cur_batch = df_batches[df_batches["batch"] == idx]

        # Read which images where used for sparse reconstruction and filter dataframe by these images
        ref_names = read_used_names(SPARSE_PATH, COLMAP)
        filt_df_cur_batch = df_cur_batch[df_cur_batch["name"].isin(ref_names)]

        # Model Aligner
        # -----------------------------------------------------------------------
        # Create output path to store (shifted) LV95 model
        GEO_REG_PATH = Path(f"{MODEL_DIR}/Model_{idx}/shifted_LV95_model")
        GEO_REG_PATH.mkdir(parents=True, exist_ok=True)

        # Save txt file of the images with the geo data (Format: image_name1.jpg X1 Y1 Z1)
        TXT_PATH = Path(f"{GEO_REG_PATH}/ref_im.txt")
        ref_im_data = ["name", "x", "y", "z"]
        df_for_txt = filt_df_cur_batch[ref_im_data]
        df_for_txt.to_csv(TXT_PATH, sep=" ", header=False, index=False)

        # Align the model with the shifted LV95 coordinates of the images
        model_aligner_command = [
            f"{COLMAP}",
            "model_aligner",
            "--input_path",
            f"{SPARSE_PATH}",
            "--output_path",
            f"{GEO_REG_PATH}",
            "--ref_images_path",
            f"{TXT_PATH}",
            "--ref_is_gps",
            "0",
            "--alignment_type",
            "ecef",
            #  (comap3.9: no command below)
            "--robust_alignment",
            "1",
            "--robust_alignment_max_error",
            "3.0",
        ]
        call(model_aligner_command)

        # Convert model to txt format
        TXT_PATH = Path(f"{MODEL_DIR}/Model_{idx}/shifted_LV95_model_txt")
        TXT_PATH.mkdir(parents=True, exist_ok=True)
        cam_pose_file = f"{TXT_PATH}/cam_extracted_m{idx}.txt"

        colmap_to_txt(GEO_REG_PATH, TXT_PATH, cam_pose_file)

    print("Done!")
