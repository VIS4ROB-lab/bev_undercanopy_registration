import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import open3d as o3d
import os
import math
import pycolmap
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from colmap_vidual import colmap_to_txt


def plot_points(points, ax, color, marker):
    xs, ys, zs = zip(*points)
    ax.scatter(xs, ys, zs, color=color, marker=marker)


def find_tree_pairs(points_source, points_target, max_dist):
    """
    This function searches for each tree in points_target the closest tree in points_source.
    If they are closer than max_dist they are considered as the same tree and added to a list of corresponding trees which is then returned.
    Parameters:
                points_source: tree positions of source
                point_target: tree positions of target (model0)
                max_dist: max searching radius to find corresponding tree
    Returns:
            corresponding_tree_indexes: list of corresponding trees indicated by their index: [source_index, target_index]

    """
    corresponding_tree_indexes = []

    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # plot_points(points_source, ax, 'b', 'o')  # blue 'o' for source points
    # plot_points(points_target, ax, 'g', '^')

    # Find closest tree
    for target_id, point_target in enumerate(points_target):
        source_id = None
        min_found_dist = math.inf

        # ax.scatter(*point_target, color='yellow', s=100, edgecolors='k', zorder=5)

        for i_source, point_source in enumerate(points_source):
            # p = ax.scatter(*point_source, color='red', s=100, edgecolors='k', zorder=5)

            dist = math.sqrt(
                (point_target[0] - point_source[0]) ** 2
                + (point_target[1] - point_source[1]) ** 2
                + (point_target[2] - point_source[2]) ** 2
            )
            if dist <= max_dist and dist < min_found_dist:
                source_id = i_source

            # plt.draw()
            # plt.pause(0.5)
            # p.remove()

        # Append indexes of found tree pair to list
        if source_id is not None:
            tree_pair = [source_id, target_id]
            corresponding_tree_indexes.append(tree_pair)

    #         ax.scatter(*points_source[source_id], color='orange', s=100, edgecolors='k', zorder=5)
    #         ax.scatter(*point_target, color='orange', s=100, edgecolors='k', zorder=5)
    #         plt.draw()
    #         plt.pause(0.5)  # Pausing to visualize the found pair
    # plt.show()

    return np.array(corresponding_tree_indexes)


# Function to select points within boundary
def select_points_within_boundary(points, boundary):
    min_x, max_x, min_y, max_y = boundary
    selected_points = []

    for point in points:
        if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
            selected_points.append(point)
    return selected_points


def read_txt_to_ply(ground_txt_file):
    ground_df = pd.read_csv(
        ground_txt_file, delimiter=" ", skiprows=3, header=None, names=["x", "y", "z", "r", "g", "b"]
    )
    x_ground = ground_df["x"].tolist()
    y_ground = ground_df["y"].tolist()
    z_ground = ground_df["z"].tolist()
    r_ground = ground_df["r"].tolist()
    g_ground = ground_df["g"].tolist()
    b_ground = ground_df["b"].tolist()
    r_norm = np.array(r_ground) / 255.0
    g_norm = np.array(g_ground) / 255.0
    b_norm = np.array(b_ground) / 255.0
    point_cloud_ground = o3d.geometry.PointCloud()
    point_cloud_ground.points = o3d.utility.Vector3dVector(list(zip(x_ground, y_ground, z_ground)))
    point_cloud_ground.colors = o3d.utility.Vector3dVector(list(zip(r_norm, g_norm, b_norm)))
    return point_cloud_ground


def rmse_translation_only(translation, src_points, target_points):
    """
    Calculate the RMSE between the translated source points and the target points.
    :param translation: Translation vector (1x3).
    :param src_points: Nx3 numpy array of source points.
    :param target_points: Nx3 numpy array of target points.
    :return: RMSE value.
    """
    translated_src_points = src_points + translation
    return np.sqrt(np.mean((translated_src_points - target_points) ** 2))


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # -                           ARGPARSE                             -
    # ------------------------------------------------------------------

    # Argparse: Create parser
    parser = argparse.ArgumentParser()
    # Argparse: Add arguments
    parser.add_argument(dest="project_path", help="Path to the project folder, containing Models/Model_i folders")
    parser.add_argument("-ver_ignore", "--ver_ignore", default=False, help="vertical_trans or not")
    parser.add_argument("-hor_ignore", "--hor_ignore", default=False, help="vertical_trans or not")
    parser.add_argument(
        "-max_tree_pair_dist", "--max_tree_pair_dist",
        default=0.5,
        help="Max distance at which corresponding tree in other model should be searched. Default: 3.0",
    )
    # Argparse: Parse arguments
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # -                           SET UP                               -
    # ------------------------------------------------------------------

    # Set max distance to default or if given to optional parser argument value
    max_dist = float(args.max_tree_pair_dist)

    PROJECT_PATH = f"{args.project_path}"
    BATCH_ALIGN_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/3_batch_align")
    BATCH_INIT_PATH = Path(f"{BATCH_ALIGN_PATH}/Init_files")
    BATCH_PATH = Path(f"{BATCH_ALIGN_PATH}/Batches")  # /Cropped_model1
    # load axes_align_transformation
    TRANS_FILE = f"{BATCH_INIT_PATH}/trans_m1_xy.txt"
    axes_align_trans_matrix = np.loadtxt(TRANS_FILE)
    inverse_trans_matrix = np.linalg.inv(axes_align_trans_matrix)

    # Create overall output folder for horizontal transformation estimation
    TRAFO_HOR = Path(f"{BATCH_ALIGN_PATH}/Horizontal_correct")
    TRAFO_HOR.mkdir(parents=True, exist_ok=True)
    TRAFO_VER = Path(f"{BATCH_ALIGN_PATH}/Vertical_correct")
    TRAFO_VER.mkdir(parents=True, exist_ok=True)

    IMAGES_LV95_BATCHES = f"{args.project_path}/Images_LV95_batches.csv"
    df_batches = pd.read_csv(IMAGES_LV95_BATCHES)
    max_batch = df_batches["batch"].max()
    nr_models = int(max_batch) + 1

    vertical_ignore = args.ver_ignore == "True"
    horizontal_ignore = args.hor_ignore == "True"

    for idx in range(1, nr_models):
        # --- Read boundaries from file
        boundaries = []
        boundary_file_path = f"{BATCH_PATH}/bounding_boxes_m{idx}.txt"
        with open(boundary_file_path, "r") as file:
            next(file)
            for line in file:
                segments = line.strip().split()
                boundary = [float(segment) for segment in segments]
                boundaries.append(boundary)

        # --- Read tree centers used for calculating transformation
        trees_eval_path = f"{BATCH_PATH}/clustered_graph_kmeans_m{idx}.csv"
        trees_eval_df = pd.read_csv(trees_eval_path, delimiter=" ")

        final_trees = []
        # Iterate over the boundaries
        for i, boundary in enumerate(boundaries):
            # ------------------------------------------------------------------
            # -----                          Grounds                       -----
            # ------------------------------------------------------------------
            ground_txt_0 = f"{BATCH_INIT_PATH}/Grounds/transformed_ground_model_0.txt"
            ground_points_0 = read_txt_to_ply(ground_txt_0)

            ground_txt_batch = f"{BATCH_PATH}/Cropped_ground{idx}/model{idx}_groundseg{i}.txt"
            ground_points_batch = read_txt_to_ply(ground_txt_batch)

            threshold = 10.0
            trans_init = np.asarray(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            )
            reg_p2p = o3d.pipelines.registration.registration_icp(
                ground_points_batch,
                ground_points_0,
                threshold,
                trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=1000),
            )
            ver_trans_matrix = reg_p2p.transformation

            if vertical_ignore:
                ver_trans_matrix = np.asarray(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
                )
            # ------------------------------------------------------------------
            # -                 Find tree pairs and minimize error             -
            # ------------------------------------------------------------------
            pc_target_tree_pts = o3d.io.read_point_cloud(
                f"{BATCH_INIT_PATH}/Tree_centers/transformed_tree_pos_model_0.ply"
            )
            pc_target_tree_pts = pc_target_tree_pts.transform(ver_trans_matrix)
            target_tree_pts_all = pc_target_tree_pts.points

            
            ## ---  using trees from graph-kmeans ---
            batch_trees = trees_eval_df[trees_eval_df["cluster_label"] == i]
            tree_coords = batch_trees[["x", "y", "z"]].values
            source_pc = o3d.geometry.PointCloud()
            source_pc.points = o3d.utility.Vector3dVector(tree_coords)

            pc_src_tree_pts = source_pc.transform(ver_trans_matrix)
            source_tree_pts = pc_src_tree_pts.points

            tree_pairs = find_tree_pairs(source_tree_pts, target_tree_pts_all, max_dist)
            src_points = np.array([source_tree_pts[src_index] for src_index, _ in tree_pairs])
            target_points = np.array([target_tree_pts_all[tgt_index] for _, tgt_index in tree_pairs])

            # ------- hor_transformation --------
            OUTPUT_SEG = Path(f"{TRAFO_HOR}/Model{idx}/Seg{i}")
            OUTPUT_SEG.mkdir(parents=True, exist_ok=True)
            print(f"Searching for horizontal transform from Seg_{i} to Model_{0}...")

            # tree_pairs = find_tree_pairs(source_tree_pts, target_tree_pts, max_dist)
            print(f"the nr of tree pairs found: {len(tree_pairs)}")
            # Need at least 3 tree pairs
            if len(tree_pairs) >= 3:
                corres_trees = o3d.utility.Vector2iVector(tree_pairs)
                ## method 1: find transformation (transl + rota + (scaling))
                trafo_est_trees = o3d.pipelines.registration.TransformationEstimationPointToPoint()  # with_scaling=True
                hor_trans_matrix_1 = trafo_est_trees.compute_transformation(
                    pc_src_tree_pts, pc_target_tree_pts, corres_trees
                )

                transformed_src_points_1 = np.dot(
                    np.hstack((src_points, np.ones((len(src_points), 1)))), hor_trans_matrix_1.T
                )[:, :3]
                rmse_method_1 = np.sqrt(np.mean(np.sum((transformed_src_points_1 - target_points) ** 2, axis=1)))

                ## method 2: consider translation only
                initial_translation = np.array([0, 0, 0])
                result = minimize(
                    rmse_translation_only, initial_translation, args=(src_points, target_points)
                )  # Minimize the RMSE
                optimal_translation = result.x
                hor_trans_matrix_2 = np.identity(4)
                hor_trans_matrix_2[0:3, 3] = optimal_translation

                rmse_method_2 = rmse_translation_only(optimal_translation, src_points, target_points)
                # Compare RMSE and choose transformation
                if rmse_method_1 < rmse_method_2:
                    print(f"rotation + translation: rmse_method_1:{rmse_method_1}<rmse_method_2:{rmse_method_2}")
                    hor_trans_matrix = hor_trans_matrix_1
                else:
                    print(f"translation: rmse_method_2:{rmse_method_2} < rmse_method_1:{rmse_method_1}")
                    hor_trans_matrix = hor_trans_matrix_2
            else:
                hor_trans_matrix = np.asarray(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
                )

            if horizontal_ignore:
                hor_trans_matrix = np.asarray(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
                )

            # --- save & trans
            trafo_path = f"{OUTPUT_SEG}/transformation_seg{i}.txt"
            np.savetxt(trafo_path, ver_trans_matrix @ hor_trans_matrix)

            tree_coords_homogeneous = np.hstack((tree_coords, np.ones((len(tree_coords), 1))))
            transformed_points_homogeneous = np.dot(
                np.dot(np.dot(tree_coords_homogeneous, ver_trans_matrix.T), hor_trans_matrix.T), inverse_trans_matrix.T
            )
            transformed_points = transformed_points_homogeneous[:, :3]
            with open(f"{OUTPUT_SEG}/tree_centers{i}.csv", "w") as f:
                f.write("tree_idx x y z\n")
                for j, coords in enumerate(transformed_points):
                    tree_idx = batch_trees.iloc[j]["tree_idx"]
                    f.write(f"{tree_idx} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}\n")
            # entire
            for j, coords in enumerate(transformed_points):
                tree_idx = batch_trees.iloc[j]["tree_idx"]
                final_trees.append([tree_idx, coords[0], coords[1], coords[2]])

            # transform batch models
            INPUT_SEG_MODEL = f"{BATCH_ALIGN_PATH}/Batches/Cropped_model{idx}/model{idx}_segment{i}"
            OUTPUT_SEG_MODEL = Path(f"{OUTPUT_SEG}/Seg{i}_Model")
            OUTPUT_SEG_MODEL.mkdir(parents=True, exist_ok=True)
            OUTPUT_SEG_MODEL_TXT = Path(f"{OUTPUT_SEG}/Seg{i}_Model_txt")
            OUTPUT_SEG_MODEL_TXT.mkdir(parents=True, exist_ok=True)
            # transform
            # trans_matrix = gt_trans_matrix @ hor_trans_matrix @ inverse_trans_matrix
            trans_matrix = ver_trans_matrix @ hor_trans_matrix @ inverse_trans_matrix
            model = pycolmap.Reconstruction(INPUT_SEG_MODEL)
            model.transform(pycolmap.SimilarityTransform3(trans_matrix[:3]))
            model.write(OUTPUT_SEG_MODEL)
            # txt model
            camera_pose_extract_file = f"{OUTPUT_SEG_MODEL_TXT}/extracted_cam{i}.txt"
            colmap_to_txt(OUTPUT_SEG_MODEL, OUTPUT_SEG_MODEL_TXT, camera_pose_extract_file)

        # save trees for the entire model
        final_trees_array = np.array(final_trees)
        np.savetxt(
            f"{TRAFO_HOR}/tree_centers_m{idx}.csv",
            final_trees_array,
            delimiter=" ",
            header="tree_idx x y z",
            fmt=["%d", "%.6f", "%.6f", "%.6f"],
            comments="",
        )
