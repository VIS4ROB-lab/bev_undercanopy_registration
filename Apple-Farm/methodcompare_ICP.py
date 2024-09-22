import open3d as o3d
import numpy as np
import copy
import pycolmap
from pathlib import Path
import pandas as pd
from subprocess import call
from scipy.spatial.transform import Rotation as R
import shutil


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996],
    )


def load_o3d_point_cloud(filename, boundary):
    data = np.loadtxt(filename)
    point = data[:, :3]
    color = data[:, 3:6] / 255.0

    xmin, xmax, ymin, ymax = boundary
    mask = (point[:, 0] >= xmin) & (point[:, 0] <= xmax) & (point[:, 1] >= ymin) & (point[:, 1] <= ymax)
    filtered_point = point[mask]
    filtered_color = color[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_point)
    pcd.colors = o3d.utility.Vector3dVector(filtered_color)
    return pcd



PROJECT_PATH = f"case1/project_views"
idx = 1
# target = 1; source = 2
point_cloud_path_1_txt = f"{PROJECT_PATH}/Correctness_loop/evaluation/comparison_method/model0.asc"
point_cloud_path_2_txt = f"{PROJECT_PATH}/Correctness_loop/evaluation/comparison_method/model{idx}.asc"
# bounding area
ymax = 1e9
ymin = -1e9
xmin = -1e9
xmax = 1e9

# ---------------------------------
boundary = [xmin, xmax, ymin, ymax]
target_pcd = load_o3d_point_cloud(point_cloud_path_1_txt, boundary)
source_pcd = load_o3d_point_cloud(point_cloud_path_2_txt, boundary)

# --- ICP
o3d.visualization.draw_geometries([source_pcd, target_pcd])

trans_init = np.eye(4)  # apply shift first to decrease the thre
# trans_init[2,3]=-100 # initial shift (diff_seasons)
trans_init[2, 3] = -95  # diff_views
threshold = 10  # 0.02
reg_p2p = o3d.pipelines.registration.registration_icp(
    source_pcd,
    target_pcd,
    threshold,
    trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
)
transformation = reg_p2p.transformation

draw_registration_result(source_pcd, target_pcd, transformation)
# source_pcd.transform(transformation)
# o3d.visualization.draw_geometries([source_pcd, target_pcd])


# --- transform model
source_model_path = f"{PROJECT_PATH}/Correctness_loop/evaluation/gt_models/GTAlign_Init_Model_{idx}"
source_model = pycolmap.Reconstruction(source_model_path)
source_model.transform(pycolmap.SimilarityTransform3(transformation[:3]))

OUTPUT_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/comparison_method/ICP")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
source_model.write(OUTPUT_PATH)

OUTPUT_PATH_TXT = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/comparison_method/ICP_txt")
OUTPUT_PATH_TXT.mkdir(parents=True, exist_ok=True)

# save transformation
transformation_output = f"{OUTPUT_PATH_TXT}/transformation_m{idx}.txt"
np.savetxt(transformation_output, transformation)

model_converter_command = [
    "colmap",
    "model_converter",
    "--input_path",
    f"{OUTPUT_PATH}",
    "--output_path",
    f"{OUTPUT_PATH_TXT}",
    "--output_type",
    "TXT",
]
call(model_converter_command)
# Create plot.txt file to plot model in cloud compare
max_err = 2.0
df_points3D = pd.read_csv(
    f"{OUTPUT_PATH_TXT}/points3D.txt",
    sep=" ",
    comment="#",
    header=None,
    usecols=range(8),
    encoding="utf8",
    engine="python",
)
df_points3D.columns = ["POINT3D_ID", "X", "Y", "Z", "R", "G", "B", "ERROR"]
df_points3D = df_points3D.drop(df_points3D[df_points3D["ERROR"] > max_err].index)
columns_plot = ["X", "Y", "Z", "R", "G", "B"]
df_plot = df_points3D[columns_plot]
df_plot.to_csv(f"{OUTPUT_PATH_TXT}/plot.txt", sep=" ", header=False, index=False)
# camera poses
file_name = "images.txt"
images_df = pd.read_csv(f"{OUTPUT_PATH_TXT}/{file_name}", sep=" ", comment="#", header=None, usecols=range(10))
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
output_poses.to_csv(f"{OUTPUT_PATH_TXT}/cam_ICP_eval_m{idx}.txt", sep=" ", index=False, header=header)

EVALUATION_DIR = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation")
shutil.copy(
    f"{OUTPUT_PATH_TXT}/cam_ICP_eval_m{idx}.txt", f"{EVALUATION_DIR}/Camera_poses_compare/cam_eval_m{idx}_ICP.txt"
)
