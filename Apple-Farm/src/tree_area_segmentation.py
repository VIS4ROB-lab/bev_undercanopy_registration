import argparse
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import os


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # -                           ARGPARSE                             -
    # ------------------------------------------------------------------

    # Argparse: Create parser
    parser = argparse.ArgumentParser()
    # Argparse: Add arguments
    parser.add_argument(dest="project_path", help="Path to the project folder, containing Models/Model_i folders")
    parser.add_argument(
        "-tree_area_dist", "--tree_area_dist", default="6.0",
        help="The distance of segment boundary to the tree center. Default: 6.0",
    )
    parser.add_argument(
        "-removing_tree_radius", default="2.0",
        help="The radius of the areas within each tree segment where other trees will be removed. Default: 2.0",
    )
    # Argparse: Parse arguments
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # -                           SET UP                               -
    # ------------------------------------------------------------------
    # Set param
    area_dist = float(args.tree_area_dist)
    cut_radius = float(args.removing_tree_radius)

    # Models path
    MODELS_PATH = Path(f"{args.project_path}/Models")

    # Read number of batches/models
    IMAGES_LV95_BATCHES = f"{args.project_path}/Images_LV95_batches.csv"
    df_batches = pd.read_csv(IMAGES_LV95_BATCHES)
    max_batch = df_batches["batch"].max()
    nr_batches = int(max_batch) + 1

    # Initialize array to indicate if enough trees were found to do transformation estimation
    enough_trees = [False] * nr_batches

    # loop through models
    for idx in range(nr_batches):

        # load input model and read points (shifted model(GPS Geo-registered))
        INPUT_MODEL = Path(f"{MODELS_PATH}/Model_{idx}/shifted_LV95_model_txt/points3D.txt")
        point_cloud_df = pd.read_csv(
            INPUT_MODEL, delimiter=" ", comment="#", usecols=[0, 1, 2, 3, 4, 5, 6], header=None, skiprows=3
        )
        point_cloud_df.columns = ["POINT3D_ID", "X", "Y", "Z", "R", "G", "B"]
        # Extract header from the original file
        with open(INPUT_MODEL, "r") as file:
            header_lines = []
            for line in file:
                if line.startswith("#"):
                    header_lines.append(line)
                else:
                    break

        # load tree positions (3d point wrt. shifted model)
        TREE_POSITIONS = Path(f"{MODELS_PATH}/Model_{idx}/tree_positions_shifted_LV95/tree_positions.txt")
        tree_pos_exists = False
        if TREE_POSITIONS.exists():
            tree_pos_exists = True
            enough_trees[idx] = True
            tree_df = pd.read_csv(TREE_POSITIONS, delimiter=",", header=0)
            x_tree = tree_df["x"].tolist()
            y_tree = tree_df["y"].tolist()
            z_tree = tree_df["z"].tolist()

        # creat output folder
        Tree_Segment = Path(f"{MODELS_PATH}/Model_{idx}/Tree_Segmentation")
        if Tree_Segment.exists():
            shutil.rmtree(Tree_Segment)
        Tree_Segment.mkdir(parents=True, exist_ok=True)

        # ------------------------------------------------------------------
        # -                    Single tree area segmentation               -
        # ------------------------------------------------------------------

        if enough_trees[idx]:
            print(f"segmenting from Model_{idx}")
            # Loop through each tree position
            for i in range(len(x_tree)):
                x_center, y_center, z_center = x_tree[i], y_tree[i], z_tree[i]

                # Segment an area around the given tree position
                mask = (
                    (point_cloud_df["X"] > x_center - area_dist)
                    & (point_cloud_df["X"] < x_center + area_dist)
                    & (point_cloud_df["Y"] > y_center - area_dist)
                    & (point_cloud_df["Y"] < y_center + area_dist)
                )
                segmented_area_df = point_cloud_df[mask].copy()

                # Remove other trees in the selected area
                for j in range(len(x_tree)):
                    if i != j:  # Skip the current tree
                        x_other, y_other = x_tree[j], y_tree[j]
                        distance = (
                            (segmented_area_df["X"] - x_other) ** 2 + (segmented_area_df["Y"] - y_other) ** 2
                        ) ** 0.5
                        segmented_area_df = segmented_area_df[distance > cut_radius]

                # Update the third line of the header with the new number of points
                header_lines[1] = f"#   POINT3D_ID, X, Y, Z, R, G, B\n"
                header_lines[2] = f"# Number of points: {len(segmented_area_df)}\n"

                # Save the segmented area to a new file with the updated header
                with open(Tree_Segment / f"tree_{i}_segmented_area.txt", "w") as file:
                    file.writelines(header_lines)
                    segmented_area_df.to_csv(file, index=False, sep=" ", header=False, lineterminator="\n")

                print(f"finish segmentation of tree{i}")

        print(f"Model_{idx} tree segmentation completed.")

    print("Done!")
