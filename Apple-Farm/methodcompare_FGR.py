import open3d as o3d
import copy
import numpy as np
import pycolmap
from pathlib import Path
import pandas as pd
from subprocess import call
from scipy.spatial.transform import Rotation as R
import shutil


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down,
        target_down,
        source_fpfh,
        target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )
    return result


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


def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    return pcd_down, pcd_fpfh


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        zoom=0.4559,
        front=[0.6452, -0.3036, -0.7011],
        lookat=[1.9892, 2.0208, 1.8945],
        up=[-0.2779, -0.9482, 0.1556],
    )



PROJECT_PATH = f"case1/project_views"
idx = 2
# target = 1; source = 2
point_cloud_path_1_txt = f"{PROJECT_PATH}/Correctness_loop/evaluation/comparison_method/model0.asc"
point_cloud_path_2_txt = f"{PROJECT_PATH}/Correctness_loop/evaluation/comparison_method/model{idx}.asc"
# bounding area
ymax = 1e9
ymin = -1e9
xmin = -1e9
xmax = 1e9
# ---------------------
boundary = [xmin, xmax, ymin, ymax]
target_pcd = load_o3d_point_cloud(point_cloud_path_1_txt, boundary)
source_pcd = load_o3d_point_cloud(point_cloud_path_2_txt, boundary)

imitial_shift = np.eye(4)
# imitial_shift[2,3]= -100
imitial_shift[2, 3] = -95
source_pcd.transform(imitial_shift)

# ---- FGR
o3d.visualization.draw_geometries([source_pcd, target_pcd])
voxel_size = 8  # 15(diff_seasons, zigzag entire) #8(zigzag m2, diff_views) #0.05
target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)

distance_threshold = voxel_size * 0.5
result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
    source_down,
    target_down,
    source_fpfh,
    target_fpfh,
    o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold),
)
transformation_matrix = result.transformation @ imitial_shift

# ---- transform
source_pcd.transform(result.transformation)
o3d.visualization.draw_geometries([source_pcd, target_pcd])
# model
source_model_path = f"{PROJECT_PATH}/Correctness_loop/evaluation/gt_models/GTAlign_Init_Model_{idx}"
source_model = pycolmap.Reconstruction(source_model_path)
source_model.transform(pycolmap.SimilarityTransform3(transformation_matrix[:3]))

OUTPUT_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/comparison_method/FGR")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
source_model.write(OUTPUT_PATH)

OUTPUT_PATH_TXT = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/comparison_method/FGR_txt")
OUTPUT_PATH_TXT.mkdir(parents=True, exist_ok=True)

# save transformation
transformation_output = f"{OUTPUT_PATH_TXT}/transformation_m{idx}.txt"
np.savetxt(transformation_output, transformation_matrix)

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
output_poses.to_csv(f"{OUTPUT_PATH_TXT}/cam_FGR_eval_m{idx}.txt", sep=" ", index=False, header=header)

EVALUATION_DIR = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation")
shutil.copy(
    f"{OUTPUT_PATH_TXT}/cam_FGR_eval_m{idx}.txt", f"{EVALUATION_DIR}/Camera_poses_compare/cam_eval_m{idx}_FGR.txt"
)
