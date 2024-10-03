import argparse
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import open3d as o3d
import os
import math


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # -                           ARGPARSE                             -
    # ------------------------------------------------------------------

    # Argparse: Create parser
    parser = argparse.ArgumentParser()
    # Argparse: Add arguments
    parser.add_argument(dest="project_path", help="Path to the project folder, containing Models/Model_i folders")
    parser.add_argument("-tree_area_dist", "--tree_area_dist", default=6.0, help="distance to the tree center")
    parser.add_argument(
        "-removing_tree_radius", "--removing_tree_radius",
        default="2.0",
        help="The radius of the areas within each tree segment where other trees will be removed.",
    )
    parser.add_argument(
        "-boundary", "--boundary", default="1e9", help="boudary of working area (remove points that belongs to house)"
    )
    # Argparse: Parse arguments
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # -                           SET UP                               -
    # ------------------------------------------------------------------

    # Set distance of tree center to default or if given to optional parser argument value
    area_dist = float(args.tree_area_dist)
    # Eliminate other trees in the selected area
    cut_radius = float(args.removing_tree_radius)

    # Models path
    MODELS_PATH = Path(f"{args.project_path}/Models")
    INIT_TRAFO_PATH = Path(f"{args.project_path}/Correctness_loop/initial_alignment")
    EVALUATION_DIR = Path(f"{args.project_path}/Correctness_loop/evaluation")
    OUTPUT_TREE = Path(f"{INIT_TRAFO_PATH}/Output/Tree_centers")
    OUTPUT_TREE.mkdir(parents=True, exist_ok=True)

    # Read number of batches/models
    IMAGES_LV95_BATCHES = f"{args.project_path}/Images_LV95_batches.csv"
    df_batches = pd.read_csv(IMAGES_LV95_BATCHES)
    max_batch = df_batches["batch"].max()
    nr_batches = int(max_batch) + 1

    gt_tran_file = Path(f"{EVALUATION_DIR}/gt_models/GT_Model_0/INIT_TRANS_GT_m0.txt")
    gt_tran = np.loadtxt(gt_tran_file)

    # Initialize array to indicate if enough trees were found to do transformation estimation
    enough_trees = [False] * nr_batches
    # loop through models
    for idx in range(nr_batches):

        # --- load input model and read points (init_transformed model)
        INPUT_MODEL = Path(f"{INIT_TRAFO_PATH}/Output/Init_transed_models/model{idx}_txt/plot.txt")
        point_cloud_df = pd.read_csv(INPUT_MODEL, sep=" ", header=None)
        point_cloud_df.columns = ["x", "y", "z", "r", "g", "b"]

        # --- load tree positions (previously extracted)
        TREE_POSITIONS = Path(f"{MODELS_PATH}/Model_{idx}/Ground_extraction/trunk_centers.txt")
        tree_pos_exists = False
        if TREE_POSITIONS.exists():
            tree_pos_exists = True
            enough_trees[idx] = True
            tree_df = pd.read_csv(TREE_POSITIONS, delimiter=r"\s+", header=0)
            idx_tree = tree_df["tree_idx"].tolist()
            x_tree = tree_df["x"].tolist()
            y_tree = tree_df["y"].tolist()
            z_tree = tree_df["z"].tolist()
            point_cloud_tree_pos = o3d.geometry.PointCloud()
            point_cloud_tree_pos.points = o3d.utility.Vector3dVector(list(zip(x_tree, y_tree, z_tree)))

            # --- Load transformation from init ICP (ICP_swisstopo.py)
            INIT_TRAFO = Path(f"{INIT_TRAFO_PATH}/TRANSFORMATION_0/transformation_model_{idx}.txt")
            trafo_0_matrix = np.loadtxt(f"{INIT_TRAFO}")

            transformed_tree_pos = point_cloud_tree_pos.transform(trafo_0_matrix @ gt_tran)
            o3d.io.write_point_cloud(f"{OUTPUT_TREE}/init_tree_centers_m{idx}.ply", transformed_tree_pos)

            transformed_coordinates = np.asarray(transformed_tree_pos.points)
            with open(f"{OUTPUT_TREE}/init_tree_centers_m{idx}.txt", "w") as f:
                f.write("tree_idx x y z\n")
                for i, coords in zip(idx_tree, transformed_coordinates):
                    f.write(f"{i} {coords[0]:.2f} {coords[1]:.2f} {coords[2]:.2f}\n")

        # creat output folder
        Tree_Segment = Path(f"{INIT_TRAFO_PATH}/Output/Fur_Tree_Segmentation/Model_{idx}")
        Tree_Segment.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # -                 Single tree area segmentation                  -
        # ------------------------------------------------------------------

        if enough_trees[idx]:
            print(f"segmenting from Model_{idx}")

            y_boundary = float(args.boundary)
            for i, (x_center, y_center, z_center) in zip(idx_tree, transformed_coordinates):
                # Segment an area around the given tree position & remove house points if necessary
                mask = (
                    (point_cloud_df["x"] > x_center - area_dist)
                    & (point_cloud_df["x"] < x_center + area_dist)
                    & (point_cloud_df["y"] > y_center - area_dist)
                    & (point_cloud_df["y"] < y_center + area_dist)
                    & (point_cloud_df["y"] < y_boundary)
                )
                segmented_area_df = point_cloud_df[mask].copy()

                # Remove other trees in the selected area
                for j, (x_other, y_other, _) in zip(idx_tree, transformed_coordinates):
                    if i != j:  # Skip the current tree
                        distance = (
                            (segmented_area_df["x"] - x_other) ** 2 + (segmented_area_df["y"] - y_other) ** 2
                        ) ** 0.5
                        segmented_area_df = segmented_area_df[distance > cut_radius]

                # Save the segmented area to a new file with the updated header
                segmented_area_df.to_csv(
                    Path(Tree_Segment / f"tree_{i}_segmented_area.txt"), index=False, sep=" ", header=True
                )
                print(f"finish segmentation of tree{i}")

        print(f"Model_{idx} tree segmentation completed.")

    print("Done!")
