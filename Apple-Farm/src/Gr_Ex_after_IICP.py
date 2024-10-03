import argparse
from pathlib import Path
from sklearn.mixture import GaussianMixture
import shutil
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import re
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
from matplotlib.lines import Line2D
import os
import glob
from Gr_Ex_heightslice import filter_by_density, plot3d_elliptical_cylinder, ground_outlier_filter


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # -                           ARGPARSE                             -
    # ------------------------------------------------------------------

    # Argparse: Create parser
    parser = argparse.ArgumentParser()
    # Argparse: Add arguments
    parser.add_argument(dest="project_path", help="Path to the project folder, containing Models/Model_i folders")
    parser.add_argument(
        "-slice_interval", "--slice_interval", default="0.5", help="interval of each slice. Default: 0.5"
    )
    parser.add_argument(
        "-expected_mean_tolerance",
        "--expected_mean_tolerance",
        default=0.8,
        help="mean tolerance when comparing mean of fitted gaussian with tree positions from bev.\
                                Should be increased when bev is not quite accurate, esp. predicted bev",
    )
    parser.add_argument("-center_area", "--center_area", default=3, help="radius of center cylinder around tree")
    parser.add_argument(
        "-expected_trunk_radius", "--expected_trunk_radius", default=0.2, help="average radius of trunks"
    )
    # Argparse: Parse arguments
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # -                           SET UP                               -
    # ------------------------------------------------------------------

    slice_interval = float(args.slice_interval)

    # parameters setting
    min_points_per_slice = 3
    further_slice_interval = 0.4
    center_area_r = float(args.center_area)
    expected_trunk_radius = float(args.expected_trunk_radius)  # 0.4~0.5m diameter
    noise_level = 0.05  # noise level
    expected_trunk_stddev = expected_trunk_radius + noise_level  # Standard deviation in XY plane
    expected_trunk_covariance = np.array(
        [[expected_trunk_stddev**2, 0], [0, expected_trunk_stddev**2]]
    )  # Covariance matrix in XY plane
    covariance_tolerance = 0.15
    expected_mean_tolerance = float(args.expected_mean_tolerance)

    # model path
    MODELS_PATH = Path(f"{args.project_path}/Models")
    INIT_TRAFO_PATH = Path(f"{args.project_path}/Correctness_loop/initial_alignment")

    # Read number of batches/models
    IMAGES_LV95_BATCHES = f"{args.project_path}/Images_LV95_batches.csv"
    df_batches = pd.read_csv(IMAGES_LV95_BATCHES)
    max_batch = df_batches["batch"].max()
    nr_batches = int(max_batch) + 1

    for idx in range(nr_batches):
        # input & output path
        INPUT_MODEL_PATH = Path(f"{INIT_TRAFO_PATH}/Output/Fur_Tree_Segmentation/Model_{idx}")
        OUTPUT_MODEL_FOLDER = Path(f"{INIT_TRAFO_PATH}/Output/Fur_Ground_extraction/Model_{idx}")
        OUTPUT_MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

        # read tree positions for model_{idx} (previously extracted)
        TREE_POSITIONS = Path(f"{INIT_TRAFO_PATH}/Output/Tree_centers/init_tree_centers_m{idx}.txt")
        if TREE_POSITIONS.exists():
            tree_df = pd.read_csv(TREE_POSITIONS, delimiter=r"\s+", header=0)
            idx_tree = tree_df["tree_idx"].tolist()
            x_tree = tree_df["x"].tolist()
            y_tree = tree_df["y"].tolist()
            z_tree = tree_df["z"].tolist()
        # --------------------------------------------------------------------------------------------
        #                         --- For each tree segment ---
        # --------------------------------------------------------------------------------------------
        trunk_centers = []
        for txt_file in INPUT_MODEL_PATH.glob("*.txt"):
            # ------------------------------------------------------------------------
            #                         --- step1: preprocessing ---
            # ------------------------------------------------------------------------
            # --- Read the points from the .txt file
            points = []
            ground_points = []
            tree_points = []
            with open(txt_file, "r") as file:
                header_line = next(file)
                for line in file:
                    parts = line.split()
                    x = float(parts[0])
                    y = float(parts[1])
                    z = float(parts[2])
                    r = float(parts[3])
                    g = float(parts[4])
                    b = float(parts[5])
                    points.append((x, y, z, r, g, b))
            points_array = np.array(
                points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("r", "f4"), ("g", "f4"), ("b", "f4")]
            )

            # # V - Create a 3D figure
            # fig = plt.figure(figsize=(12, 10))
            # ax = fig.add_subplot(111, projection='3d')
            # # ax.set_title(f"{txt_file.stem} : Tree Segment with Fitted and Expected Gaussians")
            # ax.tick_params(axis='x', labelsize=28)  # Adjust 'large' to the desired size or use a numerical value
            # ax.tick_params(axis='y', labelsize=28)  # Adjust 'large' to the desired size or use a numerical value
            # ax.tick_params(axis='z', labelsize=28)
            # ax.set_yticks([4, 0, -4])
            # ax.view_init(elev=20, azim=20)
            # plt.ion()
            # plt.pause(1) #uncertain
            # # # V - Original points in light gray
            # # ax.scatter(points_array['x'], points_array['y'], points_array['z'], c='green', marker='o', alpha=0.1, s=1) #,c='gray'; color=(0.7, 0.7, 0.7)

            # --- Define the bounds for the outliers, (Calculate the Interquartile Range (IQR) for the z-values
            q1 = np.percentile(points_array["z"], 25)
            q3 = np.percentile(points_array["z"], 75)
            iqr = q3 - q1
            lower_bound = q1 - 1 * iqr
            upper_bound = q3 + 3 * iqr
            filtered_points = points_array[(points_array["z"] >= lower_bound) & (points_array["z"] <= upper_bound)]

            min_z = np.min(filtered_points["z"])
            max_z = np.max(filtered_points["z"])

            # --- center area segmentation using tree postion from BEV
            match = re.search(r"\d+", txt_file.stem)  # find tree_idx
            if match:
                number = int(match.group())
                index = idx_tree.index(number)
                x_center = x_tree[index]
                y_center = y_tree[index]
                z_center = z_tree[index]

                mask = (filtered_points["x"] - x_center) ** 2 + (
                    filtered_points["y"] - y_center
                ) ** 2 <= center_area_r**2
                center_area_df = filtered_points[mask].copy()
                # # # V - Points after outlier removal in darker gray
                # ax.scatter(center_area_df['x'], center_area_df['y'], center_area_df['z'], c='green', marker='.', alpha=0.5, s=8)

                # # below_threshold = center_area_df[center_area_df['z'] < -0.2]
                # # above_threshold = center_area_df[center_area_df['z'] >= -0.2]
                # # ax.scatter(below_threshold['x'], below_threshold['y'], below_threshold['z'], c='green', marker='.', alpha=0.1, s=8)
                # # ax.scatter(above_threshold['x'], above_threshold['y'], above_threshold['z'], c='green', marker='.', alpha=0.5, s=8)

                # plt.savefig(f"{OUTPUT_MODEL_FOLDER}/vidual_treeseg_init.svg", format='svg')

            # -------------------------------------------------------------------------
            #                         --- step2: slicing & gaussian fitting ---
            # -------------------------------------------------------------------------
            # Slice the tree area along the height direction
            trunk_detected = 0
            top_of_trunk = False
            outlier_points_array = None
            print("----------------------------------")
            print("start slicing and gaussian fitting")

            for z in np.arange(min_z, max_z, slice_interval):
                slice_mask = (center_area_df["z"] >= z) & (center_area_df["z"] < z + slice_interval)
                slice_points = center_area_df[slice_mask]
                # # V - Points in the current slice
                ## ax.scatter(slice_points['x'], slice_points['y'], slice_points['z'], c='black', marker='o', s=2)

                # --- preprocessing: If the number of points in the slice is too small, add the points into 'outliers', and continue to the next slice
                if len(slice_points) < min_points_per_slice:
                    if outlier_points_array is None:
                        outlier_points_array = np.array(
                            slice_points,
                            dtype=np.dtype(
                                [("x", "f4"), ("y", "f4"), ("z", "f4"), ("r", "f4"), ("g", "f4"), ("b", "f4")]
                            ),
                        )
                    else:
                        new_slice_array = np.array(
                            slice_points,
                            dtype=np.dtype(
                                [("x", "f4"), ("y", "f4"), ("z", "f4"), ("r", "f4"), ("g", "f4"), ("b", "f4")]
                            ),
                        )
                        outlier_points_array = np.concatenate((outlier_points_array, new_slice_array))
                    continue
                # --- preprocessing: remove outliers according to density in each slice
                # VDen ---
                # plot_density(slice_points['x'], slice_points['y'],z)
                candidate_trunk, candidate_ground = filter_by_density(
                    [x_center, y_center], slice_points, eps=0.5, min_samples=6
                )
                if candidate_trunk is not None:
                    filtered_points = candidate_trunk
                else:
                    filtered_points = slice_points
                # plot_density(filtered_points['x'], filtered_points['y'],z)

                # # V - Points in the current slice
                # ax.scatter(filtered_points['x'], filtered_points['y'], filtered_points['z'], c='black', marker='o', s=3)

                slice_points_for_gmm = np.vstack((filtered_points["x"], filtered_points["y"], filtered_points["z"])).T

                # --- Apply Gaussian Mixture Model to the slice
                gmm = GaussianMixture(n_components=1, covariance_type="full").fit(slice_points_for_gmm)
                # # V -  3d gaussian vidualization
                # plot3d_elliptical_cylinder(ax, gmm.means_[0], gmm.covariances_[0], slice_interval, color='red', alpha=1)

                expected_center = np.array([x_center, y_center])
                actual_z = gmm.means_[0][2]
                new_center = np.array([expected_center[0], expected_center[1], actual_z])

                # plot3d_elliptical_cylinder(ax, new_center, expected_trunk_covariance, slice_interval, color='blue', alpha=1)
                # plt.draw()
                # plt.pause(1)
                # plt.show()

                # --- trunk detection: covariance and mean checking
                mean_from_center = np.linalg.norm(gmm.means_[0][:2] - np.array([x_center, y_center]))

                if (
                    np.allclose(gmm.covariances_[0][:2, :2], expected_trunk_covariance, atol=covariance_tolerance)
                    and mean_from_center <= expected_mean_tolerance
                ):
                    trunk_points_mask = (filtered_points["z"] >= z) & (filtered_points["z"] < z + slice_interval)
                    trunk_points = filtered_points[trunk_points_mask]
                    if not trunk_detected:
                        # save the center points of end of trunks
                        trunk_center = np.concatenate([[number], gmm.means_[0][:2], [z]])
                        trunk_centers.append(trunk_center)

                        trunk_points_array = np.array(
                            trunk_points,
                            dtype=np.dtype(
                                [("x", "f4"), ("y", "f4"), ("z", "f4"), ("r", "f4"), ("g", "f4"), ("b", "f4")]
                            ),
                        )
                    else:
                        new_points_array = np.array(
                            trunk_points,
                            dtype=np.dtype(
                                [("x", "f4"), ("y", "f4"), ("z", "f4"), ("r", "f4"), ("g", "f4"), ("b", "f4")]
                            ),
                        )
                        trunk_points_array = np.concatenate((trunk_points_array, new_points_array))
                    print(f"{txt_file.stem} trunk{z} detected successfully")

                    trunk_detected = 1

                else:
                    if trunk_detected == 1:
                        top_of_trunk = True
                        canopy_points_mask = points_array["z"] >= z
                        canopy_points = points_array[canopy_points_mask]
                        canopy_points_array = np.array(
                            canopy_points,
                            dtype=np.dtype(
                                [("x", "f4"), ("y", "f4"), ("z", "f4"), ("r", "f4"), ("g", "f4"), ("b", "f4")]
                            ),
                        )
                        print(f"{txt_file.stem} canopy{z} detected successfully")
                        break  # Exit the loop once the trunk is found

            # legend_elements = [Line2D([0], [0], color='red', lw=8, label='Fitted Gaussian'),
            # Line2D([0], [0], color='blue', lw=8, label='Expected Gaussian')]
            # plt.savefig(f"{OUTPUT_MODEL_FOLDER}/vidual_treeseg_output.svg", format='svg')

            if trunk_detected == 0:
                print(f"{txt_file.stem} detected failed !!!!!!!!!!!!!!")
                continue
            left_points_mask = points_array["z"] < z
            left_points = points_array[left_points_mask]

            if outlier_points_array is not None:
                mask = ~np.isin(left_points, outlier_points_array)
                left_points = left_points[mask]

            left_points_array = np.array(
                left_points,
                dtype=np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("r", "f4"), ("g", "f4"), ("b", "f4")]),
            )

            # ---------------------------------------------------------------------
            #           --- step3: post-processing: ground extraction ---
            # ---------------------------------------------------------------------
            """After finding the trunk, continue to process the slices below the trunk to separate the base of the tree from the ground"""
            starting_points = np.copy(left_points)
            min_z = np.min(starting_points["z"])
            max_z = np.max(starting_points["z"])

            x_mean = np.mean(trunk_points_array["x"])
            y_mean = np.mean(trunk_points_array["y"])
            # # VV -
            # fig = plt.figure(figsize=(12, 10))
            # ax = fig.add_subplot(111, projection='3d')
            # ax.set_title(f"{txt_file.stem} : 3D Scatter with Fitted and Expected Gaussians")
            # ax.scatter(starting_points['x'], starting_points['y'], starting_points['z'], c='gray', marker='o', alpha=0.3, s=2)
            # plt.ion()
            # plt.pause(1)

            # ---- remove trunk & use DBSCAN to remove the outlier
            fur_trunk = np.copy(trunk_points_array)
            trunk_z_mean = np.mean(fur_trunk["z"])  # will be used to check correctness of trunk

            mask = ~np.isin(starting_points, fur_trunk)
            left_wo_trunk = starting_points[mask]
            # # VV -
            # ax.scatter(left_wo_trunk['x'], left_wo_trunk['y'], left_wo_trunk['z'], c='white', marker='o', alpha=0.3, s=2)

            fur_eps = 0.25
            outliers, ground_points = ground_outlier_filter(left_wo_trunk, trunk_z_mean, fur_eps=fur_eps)
            if outlier_points_array is None:
                outlier_points_array = np.copy(outliers)
            else:
                outlier_points_array = np.concatenate((outlier_points_array, outliers))
            # # VV -
            # ax.scatter(ground_points['x'], ground_points['y'], ground_points['z'], c='red', marker='o', alpha=0.3, s=2)
            # plt.draw()
            # plt.pause(3)
            # plt.show()

            mask1 = ~np.isin(points_array, fur_trunk)
            else_points = points_array[mask1]
            mask2 = ~np.isin(else_points, ground_points)
            else_points = else_points[mask2]

            # -------------------------------------------------------------------------------------------------
            #                         ---  save ---
            # -------------------------------------------------------------------------------------------------
            further_file_folder = Path(f"{OUTPUT_MODEL_FOLDER}/further_saperation")
            further_file_folder.mkdir(parents=True, exist_ok=True)
            fur_trunk_file_folder = Path(f"{further_file_folder}/trunk")
            fur_trunk_file_folder.mkdir(parents=True, exist_ok=True)
            ground_points_file_folder = Path(f"{further_file_folder}/ground")
            ground_points_file_folder.mkdir(parents=True, exist_ok=True)
            else_points_file_folder = Path(f"{further_file_folder}/else")
            else_points_file_folder.mkdir(parents=True, exist_ok=True)

            fur_trunk_file_path = fur_trunk_file_folder / f"trunk_{txt_file.stem}.txt"
            ground_points_file_path = ground_points_file_folder / f"ground_{txt_file.stem}.txt"
            else_points_file_path = else_points_file_folder / f"else_{txt_file.stem}.txt"

            np.savetxt(
                fur_trunk_file_path,
                fur_trunk,
                fmt="%f %f %f %f %f %f",
                header=f"# 3D point list with one line of data per point:\n"
                f"# X, Y, Z, R, G, B\n"
                f"# Number of points: {len(fur_trunk)}",
                comments="",
            )

            np.savetxt(
                ground_points_file_path,
                ground_points,
                fmt="%f %f %f %f %f %f",
                header=f"# 3D point list with one line of data per point:\n"
                f"# X, Y, Z, R, G, B\n"
                f"# Number of points: {len(ground_points)}",
                comments="",
            )

            np.savetxt(
                else_points_file_path,
                else_points,
                fmt="%f %f %f %f %f %f",
                header=f"# 3D point list with one line of data per point:\n"
                f"# X, Y, Z, R, G, B\n"
                f"# Number of points: {len(else_points)}",
                comments="",
            )
            print(f"further separation for tree{txt_file} of model{idx} completed")

        # --------------------------------------------------------------------------------------------
        #                         --- For every entire model ---
        #    integrate to 3 parts: final ground & trunk & else(canopy & outliers)
        # --------------------------------------------------------------------------------------------
        point_types_info = {
            "ground": {
                "pattern": "ground_tree_*_segmented_area.txt",
                "read_file": ground_points_file_folder,
                "output_file": f"seg_ground_model{idx}.txt",
            },
            "trunk": {
                "pattern": "trunk_tree_*_segmented_area.txt",
                "read_file": fur_trunk_file_folder,
                "output_file": f"trunk_model{idx}.txt",
            },
            "else_points": {
                "pattern": "else_tree_*_segmented_area.txt",
                "read_file": else_points_file_folder,
                "output_file": f"seg_else_model{idx}.txt",
            },
        }
        # ----------------- loop and integrate for each group -----------------
        initial_trans_file = (
            f"{args.project_path}/Correctness_loop/initial_alignment/TRANSFORMATION_0/transformation_model_{idx}.txt"
        )
        initial_aligned_trans = np.loadtxt(initial_trans_file)
        inverse_initial_trans = np.linalg.inv(initial_aligned_trans)
        # parameters setting
        tree_ground_thre_z = 0.5  # 0.6
        tree_center_thre_z = 2.5
        ground_nr_thre = 200
        ground_medians = {}
        ground_point_nr = {}
        for point_type, info in point_types_info.items():
            pattern = info["pattern"]
            file_list = glob.glob(os.path.join(info["read_file"], pattern))

            all_content = []
            total_points = 0
            tree_centers = []
            for file_path in file_list:
                single_content = []

                with open(file_path, "r") as file:
                    content = file.readlines()

                    if point_type == "ground":
                        ground_points = np.array([list(map(float, line.split())) for line in content[3:]])
                        ground_point_nr[int(Path(file_path).stem.split("_")[2])] = ground_points.size
                        if ground_points.size > 0:
                            median_z = np.median(ground_points[:, 2])
                            ground_medians[int(Path(file_path).stem.split("_")[2])] = median_z
                        else:
                            ground_medians[int(Path(file_path).stem.split("_")[2])] = None

                    if point_type == "trunk":
                        trunk_points = np.array([list(map(float, line.split())) for line in content[3:]])
                        median_ground_z = ground_medians.get(int(Path(file_path).stem.split("_")[2]))

                        if ground_point_nr[int(Path(file_path).stem.split("_")[2])] < ground_nr_thre:
                            single_content.extend(content[3:])
                            all_content.extend(content[3:])
                            total_points += int(content[2].split(":")[1].strip())
                        else:  # for winter data with enough ground points: filter out the tree centers with anomalous z value
                            if median_ground_z is None:
                                print(f"------------{np.median(z_ground)}---------")
                                if trunk_points[:, 2].min() - np.median(z_ground) < 1:  # 1.5
                                    print(f"points saved")
                                    single_content.extend(content[3:])
                                    all_content.extend(content[3:])
                                    total_points += int(content[2].split(":")[1].strip())
                            else:
                                single_content.extend(content[3:])
                                all_content.extend(content[3:])
                                total_points += int(content[2].split(":")[1].strip())
                    else:
                        single_content.extend(content[3:])
                        all_content.extend(content[3:])
                        total_points += int(content[2].split(":")[1].strip())
                # ---------------- trunk centers extraction -----------------
                if point_type == "trunk":
                    # read tree id 'trunk_tree_i_segmented_area.txt'
                    tree_id = int(Path(file_path).stem.split("_")[2])

                    if single_content:
                        trunk_points = np.array([list(map(float, point.split())) for point in single_content])
                        trunk_points_positions = trunk_points[:, :3]
                        gmm = GaussianMixture(n_components=1, covariance_type="full")
                        gmm.fit(trunk_points_positions[:, :2])  # x,y only

                        center_x, center_y = gmm.means_[0]

                        # with refinement
                        if ground_point_nr[int(Path(file_path).stem.split("_")[2])] < ground_nr_thre:
                            center_z = np.min(trunk_points_positions[:, 2])
                        else:
                            if median_ground_z is None:
                                center_z = np.min(trunk_points_positions[:, 2])
                            else:
                                if (
                                    abs(np.min(trunk_points_positions[:, 2]) - median_ground_z) < tree_ground_thre_z
                                ):  # 0.6
                                    center_z = np.min(trunk_points_positions[:, 2])  # Minimum z-value from trunk points
                                else:
                                    center_z = median_ground_z
                        # wo/ refinement
                        center_z = np.min(trunk_points_positions[:, 2])
                        tree_centers.append((tree_id, center_x, center_y, center_z))

            if point_type == "trunk":
                trunk_center_path = OUTPUT_MODEL_FOLDER / f"trunk_centers.txt"
                average_height = sum(center[3] for center in tree_centers) / len(tree_centers)
                transformed_centers = []
                with open(trunk_center_path, "w") as file:
                    file.write("tree_idx x y z\n")
                    for center in tree_centers:
                        if center[3] - average_height < tree_center_thre_z:  # 2
                            file.write(f"{int(center[0])} {center[1]:.2f} {center[2]:.2f} {center[3]:.2f}\n")
                            # co_init
                            homo_coord = np.array([center[1], center[2], center[3], 1])
                            transformed_coord = homo_coord.dot(inverse_initial_trans.T)
                            transformed_centers.append((int(center[0]), *transformed_coord[:3]))

                TREE_CENTERS_EVAL_PATH = Path(f"{args.project_path}/Correctness_loop/evaluation/Tree_centers_eval/Ours")
                TREE_CENTERS_EVAL_PATH.mkdir(parents=True, exist_ok=True)
                if idx == 0:
                    shutil.copy(trunk_center_path, f"{TREE_CENTERS_EVAL_PATH}/base_tree.txt")
                else:
                    shutil.copy(trunk_center_path, f"{TREE_CENTERS_EVAL_PATH}/initial_aligned_tree{idx}.txt")

                    co_init_trees_path = f"{TREE_CENTERS_EVAL_PATH}/colmap_init_tree{idx}.txt"
                    with open(co_init_trees_path, "w") as file:
                        file.write("tree_idx x y z\n")
                        for center in transformed_centers:
                            file.write(f"{center[0]} {center[1]:.2f} {center[2]:.2f} {center[3]:.2f}\n")

            # ---------------- ground refine -----------------
            # parameters
            if point_type == "ground":
                X_values = []
                Y_values = []
                Z_values = []
                R_values = []
                G_values = []
                B_values = []

                for line in all_content:
                    values = line.split()
                    X_values.append(float(values[0]))
                    Y_values.append(float(values[1]))
                    Z_values.append(float(values[2]))
                    R_values.append(float(values[3]))
                    G_values.append(float(values[4]))
                    B_values.append(float(values[5]))

                entire_ground = {
                    "x": np.array(X_values),
                    "y": np.array(Y_values),
                    "z": np.array(Z_values),
                    "r": np.array(R_values),
                    "g": np.array(G_values),
                    "b": np.array(B_values),
                }

                Eground = np.column_stack((entire_ground["x"], entire_ground["y"], entire_ground["z"]))
                pca = PCA(n_components=3)
                pca.fit(Eground)
                pc1 = pca.components_[0]
                normal_vector = pca.components_[2]

                projected_ground_3d = np.dot(Eground, normal_vector)
                median_height_all = np.median(projected_ground_3d)
                Eoutlier_mask = projected_ground_3d > median_height_all + 0.8

                projected_ground_2d = Eground.dot(np.column_stack((pc1, normal_vector)))
                clustering_norm = DBSCAN(eps=0.25, min_samples=10).fit(projected_ground_2d)
                labels_norm = clustering_norm.labels_

                unique_labels_norm = np.unique(labels_norm)
                label_heights = {
                    label: projected_ground_3d[labels_norm == label] for label in unique_labels_norm if label != -1
                }
                largest_cluster_label = max(
                    filter(lambda x: x != -1, unique_labels_norm), key=lambda x: len(label_heights[x])
                )
                median_height_largest_cluster = np.median(label_heights[largest_cluster_label])
                threshold = 0.6  # 0.4 # tbd
                for label in unique_labels_norm:
                    if label != -1:
                        if np.mean(label_heights[label]) - median_height_largest_cluster > threshold:
                            Eoutlier_mask[labels_norm == label] = True

                fin_outlier = pd.DataFrame(entire_ground)[Eoutlier_mask]
                fin_ground = pd.DataFrame(entire_ground)[~Eoutlier_mask]

                final_ground_path = os.path.join(OUTPUT_MODEL_FOLDER, f"ground_model{idx}_clean.txt")
                np.savetxt(
                    final_ground_path,
                    fin_ground,
                    fmt="%f %f %f %f %f %f",
                    header=f"# 3D point list with one line of data per point:\n"
                    f"# X, Y, Z, R, G, B\n"
                    f"# Number of points: {len(fin_ground)}",
                    comments="",
                )
                z_ground = fin_ground["z"].to_numpy()

            # ------------------- end --------------------------

            integrate_file_path = os.path.join(OUTPUT_MODEL_FOLDER, info["output_file"])
            with open(integrate_file_path, "w") as output_file:
                # Write the updated header to the output file
                output_file.write("# 3D point list with one line of data per point:\n")
                output_file.write("# X, Y, Z, R, G, B\n")
                output_file.write(f"# Number of points: {total_points}\n")

                # Write the integrated content to the output file
                output_file.writelines(all_content)

        print(f"model{idx} completed")
    print("Done!")


# TODO: adjust the orientation of ground (normal) and trunks, extract again and combine the result
# TODO: for the trunk centers refinement: use the terrain right below the trunks (set a center area)
