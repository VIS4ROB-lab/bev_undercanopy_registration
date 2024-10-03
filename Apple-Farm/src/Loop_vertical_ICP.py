import argparse
import pandas as pd
from pathlib import Path
import numpy as np
import csv
import open3d as o3d
import os
import shutil
import pycolmap
from subprocess import call
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon, Point
import re
import copy
from colmap_vidual import colmap_to_txt


def read_point_cloud_2d(path):
    point_cloud = o3d.io.read_point_cloud(path)
    points_2d = [[point[0], point[1]] for point in np.asarray(point_cloud.points)]
    return Polygon(points_2d).convex_hull


def extract_overlapping_points(point_cloud, overlap_polygon):
    overlapping_points = []
    for point in np.asarray(point_cloud.points):
        if overlap_polygon.contains(Point(point[0], point[1])):
            overlapping_points.append(point)
    return np.array(overlapping_points)


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


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # -                           ARGPARSE                             -
    # ------------------------------------------------------------------

    # Argparse: Create parser
    parser = argparse.ArgumentParser()
    # Argparse: Add arguments
    parser.add_argument(dest="project_path", help="Path to the project folder containing Models/Model_i folders.")
    parser.add_argument(dest="run_count", type=int, help="Current run count for the script.")
    parser.add_argument(dest="save_intervals", help="Comma-separated intervals at which to save results.")
    parser.add_argument("--icp_threshold", default="10.0", help="threshold used in the p2p icp")
    # Argparse: Parse arguments
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # -                            Set Up                              -
    # ------------------------------------------------------------------
    save_intervals = list(map(int, args.save_intervals.split(",")))
    run_count = args.run_count

    MODEL_PATH = Path(f"{args.project_path}/Models")
    PROJECT_PATH = Path(args.project_path)
    IMAGES_LV95_BATCHES = f"{args.project_path}/Images_LV95_batches.csv"

    # Read number of batches/models
    df_batches = pd.read_csv(IMAGES_LV95_BATCHES)
    max_batch = df_batches["batch"].max()
    nr_batches = int(max_batch) + 1

    # Create output folders:
    # Overall output path for icp with swisstopo
    ICP_MODELS_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/1_ICP_models")
    ICP_MODELS_PATH.mkdir(parents=True, exist_ok=True)
    # Output of ICP: transformed ground models
    GROUND_OUTPUT_PATH = Path(f"{ICP_MODELS_PATH}/Output/ground_models")
    GROUND_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    # Output of ICP: transformed trees
    V_TREE_PATH = Path(f"{ICP_MODELS_PATH}/Output/V_Tree_centers")
    V_TREE_PATH.mkdir(parents=True, exist_ok=True)
    # Output of ICP: Transformation
    TRANSFORMATION_1_PATH = Path(f"{ICP_MODELS_PATH}/TRANSFORMATION_1")
    TRANSFORMATION_1_PATH.mkdir(parents=True, exist_ok=True)

    # INPUTS:
    ICP_INPUT = Path(f"{ICP_MODELS_PATH}/Input")
    ICP_INPUT.mkdir(parents=True, exist_ok=True)
    ICP_INPUT_GROUND = Path(f"{ICP_INPUT}/ground_models")
    ICP_INPUT_GROUND.mkdir(parents=True, exist_ok=True)

    GT_INIT_MODEL_PATH = Path(f"{args.project_path}/Correctness_loop/evaluation")
    INIT_ICP_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/initial_alignment")

    # --------------------------------------------------------------------------
    # -       Inputs preparation (copy from initial_align output, txt2ply)     -
    # --------------------------------------------------------------------------
    if run_count == 1:
        INPUT_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/initial_alignment")

        all_grounds = {}
        for idx in range(nr_batches):
            # ----------------- ground --------------------
            ground_txt = f"{INPUT_PATH}/Output/Fur_Ground_extraction/Model_{idx}/ground_model{idx}_clean.txt"
            ground_df = pd.read_csv(
                ground_txt, delimiter=" ", skiprows=3, header=None, names=["x", "y", "z", "r", "g", "b"]
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
            point_cloud_ground.points = o3d.utility.Vector3dVector(np.vstack((x_ground, y_ground, z_ground)).T)
            point_cloud_ground.colors = o3d.utility.Vector3dVector(list(zip(r_norm, g_norm, b_norm)))

            if idx == 0:  # base model
                txt_copy_ground = f"{ICP_INPUT_GROUND}/fixed_ground_model_{idx}.txt"

                o3d.io.write_point_cloud(
                    os.path.join(GROUND_OUTPUT_PATH, f"fixed_ground_model_0.ply"), point_cloud_ground
                )
                shutil.copy(
                    os.path.join(GROUND_OUTPUT_PATH, f"fixed_ground_model_0.ply"),
                    f"{ICP_INPUT_GROUND}/fixed_ground_model_{idx}.ply",
                )
            else:
                txt_copy_ground = f"{ICP_INPUT_GROUND}/init_transformed_ground_model_{idx}.txt"

                o3d.io.write_point_cloud(
                    os.path.join(GROUND_OUTPUT_PATH, f"transformed_ground_model_{idx}.ply"), point_cloud_ground
                )
                shutil.copy(
                    os.path.join(GROUND_OUTPUT_PATH, f"transformed_ground_model_{idx}.ply"),
                    f"{ICP_INPUT_GROUND}/init_transformed_ground_model_{idx}.ply",
                )

            transformed_ground = np.vstack((x_ground, y_ground, z_ground, r_ground, g_ground, b_ground)).T
            transformed_ground_df = pd.DataFrame(transformed_ground, columns=["x", "y", "z", "r", "g", "b"])
            transformed_ground_df.to_csv(txt_copy_ground, index=False, sep=" ", header=True)

            # ----------------- trunk centers -------------------
            TREE_POSITIONS = Path(f"{INPUT_PATH}/Output/Fur_Ground_extraction/Model_{idx}/trunk_centers.txt")
            if TREE_POSITIONS.exists():
                shutil.copy(TREE_POSITIONS, f"{ICP_INPUT}/init_transed_tree_centers_m{idx}.txt")
                tree_df = pd.read_csv(TREE_POSITIONS, delimiter=r"\s+", header=0)
                idx_tree = tree_df["tree_idx"].tolist()
                x_tree = tree_df["x"].tolist()
                y_tree = tree_df["y"].tolist()
                z_tree = tree_df["z"].tolist()
                # save .ply
                point_cloud_tree_pos = o3d.geometry.PointCloud()
                point_cloud_tree_pos.points = o3d.utility.Vector3dVector(np.vstack((x_tree, y_tree, z_tree)).T)
                path_tree_cloud = f"{ICP_INPUT}/init_transed_tree_centers_m{idx}.ply"
                o3d.io.write_point_cloud(path_tree_cloud, point_cloud_tree_pos)
    else:
        INPUT_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/2_horizontal")

    # middle save
    if run_count in save_intervals:
        MIDDLE_PATH = Path(f"{ICP_MODELS_PATH}/middle_result_{run_count}")
        MIDDLE_PATH.mkdir(parents=True, exist_ok=True)
        MIDDLE_GROUND_PATH = Path(f"{MIDDLE_PATH}/Output/ground_models")
        MIDDLE_GROUND_PATH.mkdir(parents=True, exist_ok=True)
        MIDDLE_MODEL_PATH = Path(f"{MIDDLE_PATH}/Output/models")
        MIDDLE_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        MIDDLE_TREE_PATH = Path(f"{MIDDLE_PATH}/Output/tree_centers")
        MIDDLE_TREE_PATH.mkdir(parents=True, exist_ok=True)
        MIDDLE_TRANSFORM_PATH = Path(f"{MIDDLE_PATH}/TRANSFORMATION_1")
        MIDDLE_TRANSFORM_PATH.mkdir(parents=True, exist_ok=True)
        print(f"run time: {run_count}")

    # ------------------------------------------------------------------
    # -                     Go through models                        -
    # ------------------------------------------------------------------
    print("Calculate vertical transformations :----------------")

    for idx in range(1, nr_batches):

        # ---------------------------------------------------------- #
        #                ICP from Open3D (point-to-point)            #
        # ---------------------------------------------------------- #
        # Get Data: Ground Models
        print(f"Model_{idx}  wrt.  Model_0...")
        point_cloud_path_ground_1 = f"{GROUND_OUTPUT_PATH}/fixed_ground_model_0.ply"
        if run_count == 1:
            point_cloud_path_ground_2 = f"{GROUND_OUTPUT_PATH}/transformed_ground_model_{idx}.ply"
        else:
            point_cloud_path_ground_2 = f"{INPUT_PATH}/Output/ground_models/transformed_ground_model_{idx}.ply"

        source_model2 = o3d.io.read_point_cloud(point_cloud_path_ground_2)
        target_model1 = o3d.io.read_point_cloud(point_cloud_path_ground_1)

        threshold = float(args.icp_threshold)
        trans_init = np.asarray(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )  # np.identity(4)
        # draw_registration_result(source_model2, target_model1, trans_init)

        # ICP wrt. the base model
        print("Apply point-to-plane ICP between models...")
        # reg_p2p = o3d.pipelines.registration.registration_icp(source_model2, target_model1, threshold, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint(), o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000)) #max_iteration=2000

        o3d.geometry.PointCloud.estimate_normals(
            target_model1,
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30),  # Adjust parameters as needed
        )
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_model2,
            target_model1,
            threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )

        transformation_matrix = reg_p2p.transformation
        # draw_registration_result(source_overlap_cloud, target_overlap_cloud, transformation_matrix)

        # ------------------ save ----------------------
        # -- Apply transformation to source (ground Model):
        transformed_source = source_model2.transform(transformation_matrix)
        o3d.io.write_point_cloud(
            f"{GROUND_OUTPUT_PATH}/vertical_transformed_ground_model_{idx}.ply", transformed_source
        )
        if run_count in save_intervals:
            o3d.io.write_point_cloud(
                f"{MIDDLE_GROUND_PATH}/vertical_transformed_ground_model_{idx}.ply", transformed_source
            )
        # save it into ,txt
        # Extract points and colors
        points_array = np.asarray(transformed_source.points)
        colors_array = np.asarray(transformed_source.colors)
        transformed_source_data = {
            "x": points_array[:, 0],
            "y": points_array[:, 1],
            "z": points_array[:, 2],
            "r": colors_array[:, 0] * 255,
            "g": colors_array[:, 1] * 255,
            "b": colors_array[:, 2] * 255,
        }
        transformed_source_df = pd.DataFrame(transformed_source_data)
        transformed_source_txt_path = f"{GROUND_OUTPUT_PATH}/vertical_transformed_ground_model_{idx}.txt"
        transformed_source_df.to_csv(transformed_source_txt_path, index=False, sep=" ", header=True)
        if run_count in save_intervals:
            transformed_source_txt_path = f"{MIDDLE_GROUND_PATH}/vertical_transformed_ground_model_{idx}.txt"
            transformed_source_df.to_csv(transformed_source_txt_path, index=False, sep=" ", header=True)

        # -- Transform tree centers
        if run_count == 1:
            INPUT_TREE = f"{ICP_INPUT}/init_transed_tree_centers_m{idx}.ply"
            INPUT_TREE_TXT = f"{ICP_INPUT}/init_transed_tree_centers_m{idx}.txt"
        else:
            INPUT_TREE = (
                f"{PROJECT_PATH}/Correctness_loop/2_horizontal/Output/Tree_centers/transformed_tree_pos_model_{idx}.ply"
            )
            INPUT_TREE_TXT = (
                f"{PROJECT_PATH}/Correctness_loop/2_horizontal/Output/Tree_centers/transformed_tree_pos_model_{idx}.txt"
            )
        TREE_CENTER = f"{V_TREE_PATH}/v_transed_tree_centers_m{idx}.ply"
        init_tree_pos = o3d.io.read_point_cloud(INPUT_TREE)
        v_tree_pos_transformed = init_tree_pos.transform(transformation_matrix)
        o3d.io.write_point_cloud(TREE_CENTER, v_tree_pos_transformed)
        transformed_coordinates = np.asarray(v_tree_pos_transformed.points)

        tree_df = pd.read_csv(INPUT_TREE_TXT, delimiter=r"\s+", header=0)
        idx_tree = tree_df["tree_idx"].tolist()
        with open(f"{V_TREE_PATH}/v_transed_tree_centers_m{idx}.txt", "w") as f:
            f.write("tree_idx x y z\n")
            for index, coords in zip(idx_tree, transformed_coordinates):
                f.write(f"{index} {coords[0]:.2f} {coords[1]:.2f} {coords[2]:.2f}\n")
        if run_count in save_intervals:
            o3d.io.write_point_cloud(f"{MIDDLE_TREE_PATH}/v_transed_tree_centers_m{idx}.ply", v_tree_pos_transformed)
            shutil.copy(
                f"{V_TREE_PATH}/v_transed_tree_centers_m{idx}.txt",
                f"{MIDDLE_TREE_PATH}/v_transed_tree_centers_m{idx}.txt",
            )

        # -- Save Transformation to file
        transformation_model = f"{TRANSFORMATION_1_PATH}/vertical_transformation_model_{idx}.txt"
        # transformation_matrix = reg_p2p.transformation
        np.savetxt(transformation_model, transformation_matrix)

        if run_count == 1:
            OVERALL_TRANSF_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/Transformation")
            OVERALL_TRANSF_PATH.mkdir(parents=True, exist_ok=True)
            overall_transformation = f"{OVERALL_TRANSF_PATH}/final_transformation_m{idx}.txt"
            overall_trans = transformation_matrix
            np.savetxt(overall_transformation, transformation_matrix)
        else:
            OVERALL_TRAFO = Path(
                f"{PROJECT_PATH}/Correctness_loop/evaluation/Transformation/final_transformation_m{idx}.txt"
            )
            final_trafo = np.loadtxt(f"{OVERALL_TRAFO}")
            # overall_trans = final_trafo @ transformation_matrix
            overall_trans = transformation_matrix @ final_trafo
            np.savetxt(OVERALL_TRAFO, overall_trans)

        if run_count in save_intervals:
            transformation_model = f"{MIDDLE_TRANSFORM_PATH}/vertical_transformation_model_{idx}.txt"
            np.savetxt(transformation_model, overall_trans)

        # --- transform models for middle result
        if run_count in save_intervals:
            # init_model_path = Path(f'{GT_INIT_MODEL_PATH}/gt_models/GTAlign_Init_Model_{idx}')
            init_model_path = Path(f"{INIT_ICP_PATH}/Output/Init_transed_models/model_{idx}")
            init_model = pycolmap.Reconstruction(init_model_path)

            init_model.transform(pycolmap.SimilarityTransform3(overall_trans[:3]))
            OUTPUT_TRANS_MODEL_PATH = Path(f"{MIDDLE_MODEL_PATH}/model{idx}")
            OUTPUT_TRANS_MODEL_PATH.mkdir(parents=True, exist_ok=True)
            init_model.write(OUTPUT_TRANS_MODEL_PATH)
            print("apply the alignment transformation matrix successfully")

            # Convert model to txt format
            TXT_PATH = Path(f"{MIDDLE_MODEL_PATH}/model{idx}_txt")
            TXT_PATH.mkdir(parents=True, exist_ok=True)
            cam_pose_file = f"{TXT_PATH}/cam_eval_m{idx}_middle{run_count}.txt"
            colmap_to_txt(OUTPUT_TRANS_MODEL_PATH, TXT_PATH, cam_pose_file)

            CAM_COMPARE_MIDDLE = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/Camera_poses_compare/middles")
            CAM_COMPARE_MIDDLE.mkdir(parents=True, exist_ok=True)
            shutil.copy(
                f"{TXT_PATH}/cam_eval_m{idx}_middle{run_count}.txt",
                f"{CAM_COMPARE_MIDDLE}/cam_eval_m{idx}_middle{run_count}_V.txt",
            )

    print("Done!")
