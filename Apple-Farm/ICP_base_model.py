import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import csv
import open3d as o3d
import pycolmap
from subprocess import call
import glob
import matplotlib.pyplot as plt
import shutil
from scipy.spatial.transform import Rotation as R
from src.colmap_vidual import colmap_to_txt


def read_points(file_path):
    points = []
    with open(file_path, "r") as file:
        for line in file:
            if not line.startswith("#"):
                _, x, y, z = line.split()
                points.append([float(x), float(y), float(z)])
    return np.array(points)


def principal_axis(points, angle_threshold_deg=20):
    centered_points = points - np.mean(points, axis=0)  # Center the points around the origin
    cov_matrix = np.cov(centered_points, rowvar=False)  # covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    principal_axis = eigenvectors[
        :, np.argmax(eigenvalues)
    ]  # Extract the principal axis (eigenvector with the largest eigenvalue)

    # filter out
    z_axis = np.array([0, 0, 1])
    angle_rad = np.arccos(np.dot(principal_axis, z_axis) / (np.linalg.norm(principal_axis) * np.linalg.norm(z_axis)))
    angle_deg = np.degrees(angle_rad)
    if angle_deg <= angle_threshold_deg:
        return principal_axis
    else:
        return None


# Function to compute the rotation matrix to align a vector with the z-axis
def rotation_matrix_from_vectors(vec1, vec2):
    """Find the rotation matrix that aligns vec1 to vec2"""
    a, b = vec1 / np.linalg.norm(vec1), vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s**2))
    return rotation_matrix


def plot_vector(ax, vector, origin=[0, 0, 0], color="blue"):
    ax.quiver(*origin, *vector, color=color, arrow_length_ratio=0.1)


def normal_vidual(average_axis, z_axis, rotation_matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the original average_axis and z_axis
    plot_vector(ax, average_axis, color="red", origin=[0, 0, 0])
    plot_vector(ax, z_axis, color="green", origin=[0, 0, 0])

    # Apply the rotation matrix to the average_axis
    transformed_average_axis = rotation_matrix @ average_axis

    plot_vector(ax, transformed_average_axis, color="blue", origin=[0, 0, 0])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title("Vector Visualization")

    plt.show()


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # -                           ARGPARSE                             -
    # ------------------------------------------------------------------

    # Argparse: Create parser
    parser = argparse.ArgumentParser()
    # Argparse: Add arguments
    parser.add_argument(dest="project_path", help="Path to the project folder containing Models/Model_i folders.")
    # Argparse: Parse arguments
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # -                            Set Up                              -
    # ------------------------------------------------------------------

    MODEL_PATH = Path(f"{args.project_path}/Models")
    PROJECT_PATH = Path(args.project_path)
    IMAGES_LV95_BATCHES = f"{args.project_path}/Images_LV95_batches.csv"
    EVALUATION_DIR = Path(f"{args.project_path}/Correctness_loop/evaluation")

    # Create output folders:
    # Overall output path for icp with swisstopo
    INIT_ICP_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/initial_alignment")
    INIT_ICP_PATH.mkdir(parents=True, exist_ok=True)

    # Input
    INITIAL_ICP_INPUT = Path(f"{INIT_ICP_PATH}/Input")
    INITIAL_ICP_INPUT.mkdir(parents=True, exist_ok=True)

    # Intermediate step: z-shift path
    Z_SHIFT_PATH = Path(f"{INIT_ICP_PATH}/z_shift")
    Z_SHIFT_PATH.mkdir(parents=True, exist_ok=True)

    # Create txt file to save shift values (only for evaluation purpose)
    z_shifts_txt_path = f"{Z_SHIFT_PATH}/applied_z_shifts.txt"
    with open(z_shifts_txt_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "z_shift"])

    # Intermediate step: z shifted ground models
    Z_SHIFTED_INPUT_PATH = Path(f"{Z_SHIFT_PATH}/ground_models")
    Z_SHIFTED_INPUT_PATH.mkdir(parents=True, exist_ok=True)

    # Output of ICP: transformed models
    TRANSFORMED_MODELS_PATH = Path(f"{INIT_ICP_PATH}/Output/Init_transed_models")
    TRANSFORMED_MODELS_PATH.mkdir(parents=True, exist_ok=True)

    # Output of ICP: Transformation (including z shift)
    TRANSFORMATION_0_PATH = Path(f"{INIT_ICP_PATH}/TRANSFORMATION_0")
    TRANSFORMATION_0_PATH.mkdir(parents=True, exist_ok=True)

    # Read number of batches/models
    df_batches = pd.read_csv(IMAGES_LV95_BATCHES)
    max_batch = df_batches["batch"].max()
    nr_batches = int(max_batch) + 1

    gt_tran_file = Path(f"{EVALUATION_DIR}/gt_models/GT_Model_0/INIT_TRANS_GT_m0.txt")
    gt_tran = np.loadtxt(gt_tran_file)
    
    # --- BASE MODEL
    BASE_GROUND_PATH = f"{MODEL_PATH}/Model_0/Ground_extraction/ground_model0_clean.txt"
    base_ground_df = pd.read_csv(
        BASE_GROUND_PATH, delimiter=" ", skiprows=3, header=None, names=["POINT3D_ID", "x", "y", "z"]
    )
    # gt coord
    homogeneous_ground_base = np.hstack((base_ground_df[["x", "y", "z"]].values, np.ones((len(base_ground_df), 1))))
    transformed_ground_base_homogeneous = homogeneous_ground_base.dot(gt_tran.T)
    transformed_ground_base = transformed_ground_base_homogeneous[:, :3]
    transformed_ground_base_df = pd.DataFrame(transformed_ground_base, columns=["x", "y", "z"])
    transformed_ground_base_df.to_csv(f"{INITIAL_ICP_INPUT}/gt_aligned_ground_m0.csv", index=False, header=True)
    x_base = transformed_ground_base_homogeneous[:, 0]
    y_base = transformed_ground_base_homogeneous[:, 1]
    z_base = transformed_ground_base_homogeneous[:, 2]

    point_cloud_path_base = f"{INITIAL_ICP_INPUT}/gt_aligned_ground_m0.ply"
    point_cloud_base = o3d.geometry.PointCloud()
    point_cloud_base.points = o3d.utility.Vector3dVector(list(zip(x_base, y_base, z_base)))
    o3d.io.write_point_cloud(point_cloud_path_base, point_cloud_base)

    destination_folder = Path(f"{TRANSFORMED_MODELS_PATH}/model_0")
    destination_folder_txt = Path(f"{TRANSFORMED_MODELS_PATH}/model0_txt")
    shutil.copytree(f"{EVALUATION_DIR}/gt_models/GT_Model_0", destination_folder, dirs_exist_ok=True)
    shutil.copytree(f"{EVALUATION_DIR}/gt_models/GT_Model_0_txt", destination_folder_txt, dirs_exist_ok=True)

    np.savetxt(f"{TRANSFORMATION_0_PATH}/transformation_model_0.txt", np.eye(4))

    # --- Initial align of other models
    for idx in range(1, nr_batches):
        print(f"Model_{idx}...")

        # Get Data: ground
        GROUND_PATH = f"{MODEL_PATH}/Model_{idx}/Ground_extraction/ground_model{idx}_clean.txt"
        # GROUND_PATH = f'{MODEL_PATH}/Model_{idx}/ground_data/ground_points_clean.txt'
        ground_df = pd.read_csv(
            GROUND_PATH, delimiter=" ", skiprows=3, header=None, names=["POINT3D_ID", "x", "y", "z"]
        )
        # x_ground = ground_df['x'].tolist()
        # GT
        homogeneous_points = np.hstack((ground_df[["x", "y", "z"]].values, np.ones((len(ground_df), 1))))
        transformed_points_homogeneous = homogeneous_points.dot(gt_tran.T)
        transformed_points = transformed_points_homogeneous[:, :3]
        x_ground = transformed_points_homogeneous[:, 0]
        y_ground = transformed_points_homogeneous[:, 1]
        z_ground = transformed_points_homogeneous[:, 2]
        transformed_df = pd.DataFrame(transformed_points, columns=["x", "y", "z"])
        transformed_df.to_csv(f"{INITIAL_ICP_INPUT}/gt_aligned_ground_m{idx}.csv", index=False, header=True)

        point_cloud_path_ground = f"{INITIAL_ICP_INPUT}/gt_aligned_ground_m{idx}.ply"
        point_cloud_ground = o3d.geometry.PointCloud()
        point_cloud_ground.points = o3d.utility.Vector3dVector(list(zip(x_ground, y_ground, z_ground)))
        # point_cloud_ground.colors = o3d.utility.Vector3dVector(list(zip(r_norm, g_norm, b_norm)))
        o3d.io.write_point_cloud(point_cloud_path_ground, point_cloud_ground)

        # ------------------------------------------------------------------
        # -                       Shift z-position                         -
        # ------------------------------------------------------------------
        print("Apply rough z-shift...")

        bin_size = 0.5
        # Rough z position base model
        hist_swiss, bins_swiss = np.histogram(z_base, bins=np.arange(min(z_base), max(z_base) + bin_size, bin_size))
        max_hist_swiss_index = np.argmax(hist_swiss)
        max_hist_swiss = bins_swiss[max_hist_swiss_index] + (bin_size / 2.0)
        # Rough z position ground model
        hist_ground, bins_ground = np.histogram(
            z_ground, bins=np.arange(min(z_ground), max(z_ground) + bin_size, bin_size)
        )
        max_hist_ground_index = np.argmax(hist_ground)
        max_hist_ground = bins_ground[max_hist_ground_index] + (bin_size / 2.0)

        # Calculate needed z-shift
        z_shift = max_hist_swiss - max_hist_ground

        # --- Save z shift value to txt
        with open(z_shifts_txt_path, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([idx, z_shift])

        # Shift ground model
        shifted_z_ground = list(np.array(z_ground) + z_shift)
        # Save shifted ground model
        shifted_data = zip(x_ground, y_ground, shifted_z_ground)
        shifted_ground_file = f"{Z_SHIFTED_INPUT_PATH}/shifted_ground_model_{idx}.txt"
        with open(shifted_ground_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y", "z"])
            writer.writerows(shifted_data)
        print(f"Saved shifted ground model at {shifted_ground_file}")

        # ------------------------------------------------- #
        #             ICP from Open3D (point-to-point)      #
        # ------------------------------------------------- #
        source = o3d.io.read_point_cloud(point_cloud_path_ground)
        target = o3d.io.read_point_cloud(point_cloud_path_base)
        o3d.visualization.draw_geometries([source, target])

        threshold = 1
        trans_init = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, z_shift], [0.0, 0.0, 0.0, 1.0]]
        )

        print("Apply point-to-point ICP...")
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source,
            target,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000),
        )
        o3d.visualization.draw_geometries([source.transform(reg_p2p.transformation), target])

        # ---------- Save Transformation to file ------
        transformation_model = f"{TRANSFORMATION_0_PATH}/transformation_model_{idx}.txt"
        transformation_matrix = reg_p2p.transformation  # homogeneous_transformation_matrix
        np.savetxt(transformation_model, transformation_matrix)
        print(f"Transformation matrix at: {transformation_model}")

        # ------------------------------------------------------------------
        # -         Transform gt_aligned -> Initial_Aligned_models         -
        # ------------------------------------------------------------------
        # ---- Apply transformation to sparse (LV95 to GT) models ----
        shifted_model_path = f"{EVALUATION_DIR}/gt_models/GTAlign_Init_Model_{idx}"
        shifted_model = pycolmap.Reconstruction(shifted_model_path)

        shifted_model.transform(pycolmap.SimilarityTransform3(transformation_matrix[:3]))

        OUTPUT_TRANS_MODEL_PATH = Path(f"{TRANSFORMED_MODELS_PATH}/model_{idx}")
        OUTPUT_TRANS_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        shifted_model.write(OUTPUT_TRANS_MODEL_PATH)
        # txt
        TXT_PATH = Path(f"{TRANSFORMED_MODELS_PATH}/model{idx}_txt")
        TXT_PATH.mkdir(parents=True, exist_ok=True)
        cam_pose_file = f"{TXT_PATH}/cam_eval_m{idx}_initial_aligned.txt"
        colmap_to_txt(OUTPUT_TRANS_MODEL_PATH, TXT_PATH, cam_pose_file)
        shutil.copy(
            f"{TXT_PATH}/cam_eval_m{idx}_initial_aligned.txt",
            f"{EVALUATION_DIR}/Camera_poses_compare/cam_eval_m{idx}_initial_aligned.txt",
        )

    print("init ICP Done!")
