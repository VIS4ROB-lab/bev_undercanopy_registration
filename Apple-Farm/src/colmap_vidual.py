from subprocess import call
import pandas as pd
from scipy.spatial.transform import Rotation as R
import numpy as np


def colmap_to_txt(INPUT_PATH, OUTPUT_PATH, extracted_camera_file):
    model_converter_command = [
        "colmap",
        "model_converter",
        "--input_path",
        f"{INPUT_PATH}",
        "--output_path",
        f"{OUTPUT_PATH}",
        "--output_type",
        "TXT",
    ]
    call(model_converter_command)

    # Create plot.txt file to plot model in cloud compare
    max_err = 2.0
    df_points3D = pd.read_csv(
        f"{OUTPUT_PATH}/points3D.txt", sep=" ", comment="#", header=None, usecols=range(8), engine="python"
    )
    df_points3D.columns = ["POINT3D_ID", "X", "Y", "Z", "R", "G", "B", "ERROR"]
    df_points3D = df_points3D.drop(df_points3D[df_points3D["ERROR"] > max_err].index)
    columns_plot = ["X", "Y", "Z", "R", "G", "B"]
    df_plot = df_points3D[columns_plot]
    df_plot.to_csv(f"{OUTPUT_PATH}/plot.txt", sep=" ", header=False, index=False)

    # camera poses
    file_name = "images.txt"
    images_df = pd.read_csv(
        f"{OUTPUT_PATH}/{file_name}", sep=" ", comment="#", header=None, usecols=range(10), engine="python"
    )
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

    output_poses = pd.concat(
        [images_df["NAME"], images_df["QW"], images_df["QX"], images_df["QY"], images_df["QZ"], camera_centers_df],
        axis=1,
    )
    header = ["filename", "qw", "qx", "qy", "qz", "x", "y", "z"]
    output_poses.to_csv(f"{extracted_camera_file}", sep=" ", index=False, header=header)
