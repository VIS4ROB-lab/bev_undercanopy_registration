import argparse
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import open3d as o3d
import os
import math
from scipy.spatial import cKDTree
from scipy.stats import median_abs_deviation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import pycolmap
from subprocess import call
from scipy.spatial.transform import Rotation as R
import re
from src.colmap_vidual import colmap_to_txt


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
    corresponding_distance = []

    # Find closest tree
    for target_id, point_target in enumerate(points_target):
        source_id = None
        min_found_dist = math.inf

        for i_source, point_source in enumerate(points_source):
            dist = math.sqrt(
                (point_target[0] - point_source[0]) ** 2
                + (point_target[1] - point_source[1]) ** 2
                + (point_target[2] - point_source[2]) ** 2
            )
            if dist <= max_dist and dist < min_found_dist:
                source_id = i_source
                min_found_dist = dist

        # Append indexes of found tree pair to list
        if source_id is not None:
            tree_pair = [source_id, target_id]
            corresponding_tree_indexes.append(tree_pair)
            corresponding_distance.append(min_found_dist)
    # TODO: filter
    return np.array(corresponding_tree_indexes)


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # -                           ARGPARSE                             -
    # ------------------------------------------------------------------

    # Argparse: Create parser
    parser = argparse.ArgumentParser()
    # Argparse: Add arguments
    parser.add_argument(dest="project_path", help="Path to the project folder, containing Models/Model_i folders")
    parser.add_argument(dest="run_count", type=int, help="Current run count for the script.")
    parser.add_argument(dest="save_intervals", help="Comma-separated intervals at which to save results.")
    parser.add_argument("scaling", type=bool, default=True, help="transformation calculation: scaling or not")

    parser.add_argument(
        "-max_tree_pair_dist",
        "--max_tree_pair_dist",
        default="4.0",
        help="Max distance at which corresponding tree in other model should be searched. Default: 3.0",
    )
    # Argparse: Parse arguments
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # -                           SET UP                               -
    # ------------------------------------------------------------------
    save_intervals = list(map(int, args.save_intervals.split(",")))
    run_count = args.run_count
    use_scaling = args.scaling
    # Set max distance
    max_dist = float(args.max_tree_pair_dist)

    IMAGES_LV95_BATCHES = f"{args.project_path}/Images_LV95_batches.csv"
    # Path to transformations from vertical correctness (ICP_swisstopo.py & ICP_diffseasonmodels.py)
    INIT_TRAFO_PATH = Path(f"{args.project_path}/Correctness_loop/initial_alignment/TRANSFORMATION_0")
    TRAFO_PATH = Path(f"{args.project_path}/Correctness_loop/1_ICP_models/TRANSFORMATION_1")

    # Models path
    MODELS_PATH = Path(f"{args.project_path}/Models")
    TRAFO_INIT = Path(f"{args.project_path}/Correctness_loop/initial_alignment")
    GT_INIT_MODEL_PATH = Path(f"{args.project_path}/Correctness_loop/evaluation")
    V_TREE_PATH = Path(f"{args.project_path}/Correctness_loop/1_ICP_models/Output/V_Tree_centers")
    ICP_INPUT = Path(f"{args.project_path}/Correctness_loop/1_ICP_models/Input")

    # Create overall output folder for horizontal transformation estimation
    TRAFO_HOR = Path(f"{args.project_path}/Correctness_loop/2_horizontal")
    TRAFO_HOR.mkdir(parents=True, exist_ok=True)

    # Create Input folder
    INPUT = Path(f"{TRAFO_HOR}/Input")
    INPUT.mkdir(parents=True, exist_ok=True)

    # Create Output folder
    OUTPUT = Path(f"{TRAFO_HOR}/Output")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    OUTPUT_GROUND = Path(f"{OUTPUT}/ground_models")
    OUTPUT_GROUND.mkdir(parents=True, exist_ok=True)
    OUTPUT_TREE = Path(f"{OUTPUT}/Tree_centers")
    OUTPUT_TREE.mkdir(parents=True, exist_ok=True)

    # Create TRANSFORMATION_2 folder
    TRAFO_2 = Path(f"{TRAFO_HOR}/TRANSFORMATION_2")
    TRAFO_2.mkdir(parents=True, exist_ok=True)

    if run_count in save_intervals:
        MIDDLE_PATH = Path(f"{TRAFO_HOR}/middle_result_{run_count}")
        MIDDLE_PATH.mkdir(parents=True, exist_ok=True)
        MIDDLE_INPUT = Path(f"{MIDDLE_PATH}/Input")
        MIDDLE_INPUT.mkdir(parents=True, exist_ok=True)
        MIDDLE_OUTPUT = Path(f"{MIDDLE_PATH}/Output")
        MIDDLE_OUTPUT.mkdir(parents=True, exist_ok=True)
        MIDDLE_OUTPUT_GROUND = Path(f"{MIDDLE_PATH}/Output/ground_models")
        MIDDLE_OUTPUT_GROUND.mkdir(parents=True, exist_ok=True)
        MIDDLE_OUTPUT_MODEL = Path(f"{MIDDLE_PATH}/Output/models")
        MIDDLE_OUTPUT_MODEL.mkdir(parents=True, exist_ok=True)
        MIDDLE_TRAFO = Path(f"{MIDDLE_PATH}/TRANSFORMATION_2")
        MIDDLE_TRAFO.mkdir(parents=True, exist_ok=True)

    # Read number of batches/models
    df_batches = pd.read_csv(IMAGES_LV95_BATCHES)
    max_batch = df_batches["batch"].max()
    nr_batches = int(max_batch) + 1

    # Initialize array to indicate if enough trees were found to do transformation estimation
    enough_trees = [False] * nr_batches
    print(" horizontal correctness --------------------------------")

    pc_target_tree_pts = o3d.io.read_point_cloud(f"{ICP_INPUT}/init_transed_tree_centers_m0.ply")
    target_tree_pts = pc_target_tree_pts.points
    # loop through models
    for idx in range(1, nr_batches):

        TREE_POSITIONS_2 = f"{V_TREE_PATH}/v_transed_tree_centers_m{idx}.ply"
        print(f"Searching for horizontal transform from Model_{idx} to Model_{0}...")

        if Path(TREE_POSITIONS_2).exists():
            enough_trees[idx] = True
            transformed_tree_pos_2 = o3d.io.read_point_cloud(TREE_POSITIONS_2)
            # ------------------------------------------------------------------
            # -                 Find tree pairs and minimize error             -
            # ------------------------------------------------------------------
            # Find pairs of corresponding trees (find closest trees withing max_dist)
            source_tree_pts = transformed_tree_pos_2.points

            tree_pairs = find_tree_pairs(source_tree_pts, target_tree_pts, max_dist)
            print(f"the nr of tree pairs found: {len(tree_pairs)}")
            # Need at least 3 tree pairs
            if len(tree_pairs) >= 3:
                corres_trees = o3d.utility.Vector2iVector(tree_pairs)
                # Use tree pairs to minimize positioning error between them --> find transformation
                trafo_est_trees = o3d.pipelines.registration.TransformationEstimationPointToPoint(
                    with_scaling=use_scaling
                )
                trafo_tree_pos = trafo_est_trees.compute_transformation(
                    transformed_tree_pos_2, pc_target_tree_pts, corres_trees
                )

                # Save hor_trafo
                trafo_path = f"{TRAFO_2}/transformation_model_{idx}.txt"
                np.savetxt(trafo_path, trafo_tree_pos)
            else:
                print(f"Not find enough tree pairs for Model_{idx}")
                enough_trees[idx] = False
        else:
            print(f"No trees for Model_{idx}")

        # ------------------------------------------------------------------
        # -                             Save                               -
        # ------------------------------------------------------------------
        # Transform with the found transformations:
        if enough_trees[idx]:

            # --Load transformation
            final_trafo_path = f"{TRAFO_2}/transformation_model_{idx}.txt"
            final_trafo = np.loadtxt(final_trafo_path)

            OVERALL_TRAFO = Path(
                f"{args.project_path}/Correctness_loop/evaluation/Transformation/final_transformation_m{idx}.txt"
            )
            overall_trafo = np.loadtxt(f"{OVERALL_TRAFO}")
            # overall_trans = overall_trafo @ final_trafo
            overall_trans = final_trafo @ overall_trafo
            np.savetxt(OVERALL_TRAFO, overall_trans)

            # --Transform tree positions
            output_tree_pos_path = f"{OUTPUT_TREE}/transformed_tree_pos_model_{idx}.ply"
            pc_tree_pos = o3d.io.read_point_cloud(f"{V_TREE_PATH}/v_transed_tree_centers_m{idx}.ply")
            pc_tree_pos_transformed = pc_tree_pos.transform(final_trafo)
            o3d.io.write_point_cloud(output_tree_pos_path, pc_tree_pos_transformed)
            if run_count in save_intervals:
                output_tree_pos_path = f"{MIDDLE_OUTPUT}/transformed_tree_pos_model_{idx}.ply"
                o3d.io.write_point_cloud(output_tree_pos_path, pc_tree_pos_transformed)
            # Saving transformed positions to .txt
            INPUT_TREE_TXT = f"{V_TREE_PATH}/v_transed_tree_centers_m{idx}.txt"
            tree_df = pd.read_csv(INPUT_TREE_TXT, delimiter=r"\s+", header=0)
            idx_tree = tree_df["tree_idx"].tolist()

            transformed_coordinates = np.asarray(pc_tree_pos_transformed.points)
            with open(f"{OUTPUT_TREE}/transformed_tree_pos_model_{idx}.txt", "w") as f:
                f.write("tree_idx x y z\n")
                for index, coords in zip(idx_tree, transformed_coordinates):
                    f.write(f"{index} {coords[0]:.2f} {coords[1]:.2f} {coords[2]:.2f}\n")
            if run_count in save_intervals:
                shutil.copy(
                    f"{OUTPUT_TREE}/transformed_tree_pos_model_{idx}.txt",
                    f"{MIDDLE_OUTPUT}/transformed_tree_pos_model_{idx}.txt",
                )

            # --Transform Ground models:
            ground_transformed_path = f"{OUTPUT_GROUND}/transformed_ground_model_{idx}.ply"
            ground = o3d.io.read_point_cloud(
                f"{args.project_path}/Correctness_loop/1_ICP_models/Output/ground_models/vertical_transformed_ground_model_{idx}.ply"
            )
            ground_transformed = ground.transform(final_trafo)
            o3d.io.write_point_cloud(ground_transformed_path, ground_transformed)
            if run_count in save_intervals:
                ground_transformed_path = f"{MIDDLE_OUTPUT_GROUND}/transformed_ground_model_{idx}.ply"
                o3d.io.write_point_cloud(ground_transformed_path, ground_transformed)
            # .txt
            ground_array = np.asarray(ground_transformed.points)
            colors_array = np.asarray(ground_transformed.colors)
            transformed_ground_data = {
                "x": ground_array[:, 0],
                "y": ground_array[:, 1],
                "z": ground_array[:, 2],
                "r": colors_array[:, 0] * 255,
                "g": colors_array[:, 1] * 255,
                "b": colors_array[:, 2] * 255,
            }
            transformed_ground_df = pd.DataFrame(transformed_ground_data)
            transformed_ground_txt_path = f"{OUTPUT_GROUND}/transformed_ground_model_{idx}.txt"
            transformed_ground_df.to_csv(transformed_ground_txt_path, index=False, sep=" ", header=True)
            if run_count in save_intervals:
                transformed_ground_txt_path = f"{MIDDLE_OUTPUT_GROUND}/transformed_ground_model_{idx}.txt"
                transformed_ground_df.to_csv(transformed_ground_txt_path, index=False, sep=" ", header=True)

            # --- transform models for middle result
            if run_count in save_intervals:
                # init_model_path = Path(f'{GT_INIT_MODEL_PATH}/gt_models/GTAlign_Init_Model_{idx}')
                init_model_path = Path(f"{TRAFO_INIT}/Output/Init_transed_models/model_{idx}")
                init_model = pycolmap.Reconstruction(init_model_path)

                init_model.transform(pycolmap.SimilarityTransform3(overall_trans[:3]))
                OUTPUT_TRANS_MODEL_PATH = Path(f"{MIDDLE_OUTPUT_MODEL}/model{idx}")
                OUTPUT_TRANS_MODEL_PATH.mkdir(parents=True, exist_ok=True)
                init_model.write(OUTPUT_TRANS_MODEL_PATH)
                print("apply the alignment transformation matrix successfully")

                # Convert model to txt format
                TXT_PATH = Path(f"{MIDDLE_OUTPUT_MODEL}/model{idx}_txt")
                TXT_PATH.mkdir(parents=True, exist_ok=True)
                cam_pose_file = f"{TXT_PATH}/cam_eval_m{idx}_middle{run_count}.txt"
                colmap_to_txt(OUTPUT_TRANS_MODEL_PATH, TXT_PATH, cam_pose_file)

                CAM_COMPARE_MIDDLE = Path(
                    f"{args.project_path}/Correctness_loop/evaluation/Camera_poses_compare/middles"
                )
                # CAM_COMPARE_MIDDLE.mkdir(parents=True, exist_ok=True)
                shutil.copy(
                    f"{TXT_PATH}/cam_eval_m{idx}_middle{run_count}.txt",
                    f"{CAM_COMPARE_MIDDLE}/cam_eval_m{idx}_middle{run_count}_H.txt",
                )

            print("transformation successfully ")

    print("Done!")
