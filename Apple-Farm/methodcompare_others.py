import numpy as np
import pycolmap
from subprocess import call
from pathlib import Path
import pandas as pd
from scipy.spatial.transform import Rotation as R
import shutil


TRANS_FILE = f" .txt" # path to the transformation matrix calculated
transformation_matrix = np.loadtxt(TRANS_FILE)


PROJECT_PATH = f"/kelly_forest_sp/case1/project_views"

idx = 2
# Transform models
MODEL_PATH = f"{PROJECT_PATH}/Correctness_loop/evaluation/gt_models/GTAlign_Init_Model_{idx}"
model = pycolmap.Reconstruction(MODEL_PATH)
model.transform(pycolmap.SimilarityTransform3(transformation_matrix[:3]))  # inverse_transformation_matrix[:3]))
OUTPUT_TRANS_MODEL_PATH = Path(
    f"{PROJECT_PATH}/Correctness_loop/evaluation/comparison_method/teaser/model{idx}_teaser_bd"
)  # teaser_bd #FRICP #fastICP
OUTPUT_TRANS_MODEL_PATH.mkdir(parents=True, exist_ok=True)
model.write(OUTPUT_TRANS_MODEL_PATH)
# TXT
TXT_PATH = Path(
    f"{PROJECT_PATH}/Correctness_loop/evaluation/comparison_method/FRICP/model{idx}_teaser_bd_txt"
)  # teaser_bd #fastICP
TXT_PATH.mkdir(parents=True, exist_ok=True)
model_converter_command = [
    "colmap",
    "model_converter",
    "--input_path",
    f"{OUTPUT_TRANS_MODEL_PATH}",
    "--output_path",
    f"{TXT_PATH}",
    "--output_type",
    "TXT",
]
call(model_converter_command)

# Create plot.txt file to plot model in cloud compare
max_err = 2.0
df_points3D = pd.read_csv(f"{TXT_PATH}/points3D.txt", sep=" ", comment="#", header=None, usecols=range(8))
df_points3D.columns = ["POINT3D_ID", "X", "Y", "Z", "R", "G", "B", "ERROR"]
df_points3D = df_points3D.drop(df_points3D[df_points3D["ERROR"] > max_err].index)
columns_plot = ["X", "Y", "Z", "R", "G", "B"]
df_plot = df_points3D[columns_plot]
df_plot.to_csv(f"{TXT_PATH}/plot.txt", sep=" ", header=False, index=False)

print(f"all in model aligned to axes")

# camera
file_name = "images.txt"
images_df = pd.read_csv(f"{TXT_PATH}/{file_name}", sep=" ", comment="#", header=None, usecols=range(10))
images_df.columns = ["IMAGE_ID", "QW", "QX", "QY", "QZ", "TX", "TY", "TZ", "CAMERA_ID", "NAME"]

images_df = images_df.drop(images_df.index[1::2]).reset_index(drop=True)

camera_centers = []
for index, row in images_df.iterrows():
    q = [row["QX"], row["QY"], row["QZ"], row["QW"]]
    t = np.array([row["TX"], row["TY"], row["TZ"]])

    # Convert quaternion to rotation matrix
    rotation = R.from_quat(q)
    rotation_matrix = rotation.as_matrix()

    # Calculate camera center in world coordinates
    camera_center = -rotation_matrix.T @ t
    camera_centers.append(camera_center)
camera_centers_df = pd.DataFrame(camera_centers, columns=["CX", "CY", "CZ"])
output = pd.concat([images_df["NAME"], camera_centers_df], axis=1)

output_poses = pd.concat(
    [images_df["NAME"], images_df["QW"], images_df["QX"], images_df["QY"], images_df["QZ"], camera_centers_df], axis=1
)
header = ["filename", "qw", "qx", "qy", "qz", "x", "y", "z"]
OUTPUT_PATH = TXT_PATH
output_poses.to_csv(f"{OUTPUT_PATH}/teaser_bd_camera_extracted_m{idx}.txt", sep=" ", index=False, header=header)
EVALUATION_DIR = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation")
shutil.copy(
    f"{OUTPUT_PATH}/teaser_bd_camera_extracted_m{idx}.txt",
    f"{EVALUATION_DIR}/Camera_poses_compare/teaser_bd_camera_extracted_m{idx}.txt",
)
