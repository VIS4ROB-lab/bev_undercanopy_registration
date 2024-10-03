import argparse
from pathlib import Path
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import re
from sklearn.cluster import DBSCAN
from scipy.spatial import distance
import os
import glob


def plot3d_elliptical_cylinder(ax, mean, covariance, height, color="blue", alpha=0.5):
    """3d gaussian vidualization"""
    # Get eigenvalues and eigenvectors of covariance matrix
    v, w = np.linalg.eigh(covariance[:2, :2])  # Only take 2x2 cov for x and y
    v = 2.0 * np.sqrt(2.0) * np.sqrt(v)
    phi = np.linspace(0, 2 * np.pi, 100)
    x = v[0] * np.cos(phi)
    y = v[1] * np.sin(phi)
    # Use eigenvectors to rotate the ellipse to its proper orientation
    xy = np.column_stack([x, y])
    x, y = np.dot(xy, w).T
    # Offset by the mean
    x += mean[0]
    y += mean[1]
    # For each z level in the height of the cylinder, plot an ellipse
    for z in [mean[2], mean[2] + height]:
        ax.plot(x, y, z, color=color)

    # Plot the sides of the cylinder connecting the ellipses
    for i in range(len(x) - 1):
        poly = [
            [x[i], y[i], mean[2]],
            [x[i + 1], y[i + 1], mean[2]],
            [x[i + 1], y[i + 1], mean[2] + height],
            [x[i], y[i], mean[2] + height],
        ]
        ax.add_collection3d(Poly3DCollection([poly], color=color, alpha=alpha))


def filter_by_density(tree_center, points_df, eps=0.5, min_samples=5):
    """Filter points based on density using DBSCAN clustering and return all significant clusters."""
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(np.column_stack((points_df["x"], points_df["y"])))
    labels = clustering.labels_
    unique_labels = np.unique(labels[labels != -1])
    clusters = sorted([points_df[labels == label] for label in unique_labels], key=len, reverse=True)
    if not clusters:
        return None, None

    # Get centroids of top 2 clusters
    centroids = [(np.mean(cluster["x"]), np.mean(cluster["y"])) for cluster in clusters[:2]]
    # Compare centroids with tree_center to determine candidate_trunk and candidate_ground
    distances = [distance.euclidean(centroid, tree_center) for centroid in centroids]
    sorted_indices = np.argsort(distances)

    candidate_trunk = clusters[sorted_indices[0]]
    candidate_ground = clusters[sorted_indices[1]] if len(clusters) > 1 else None
    return candidate_trunk, candidate_ground


def ground_outlier_filter(
    points_df, tree_z_mean, eps=0.6, min_samples=10, fur_eps=0.25, fur_min_samples=10, min_fur_ground=320
):  
    """Filter points based on density using DBSCAN clustering and return all significant clusters."""
    clustering_xz = DBSCAN(eps=eps, min_samples=min_samples).fit(np.column_stack((points_df["x"], points_df["z"])))
    labels_xz = clustering_xz.labels_

    clustering_yz = DBSCAN(eps=eps, min_samples=min_samples).fit(np.column_stack((points_df["y"], points_df["z"])))
    labels_yz = clustering_yz.labels_

    # -----  initial outlier removement -----
    outlier_mask = np.zeros(len(points_df), dtype=bool)
    initial_outliers = (labels_xz == -1) | (labels_yz == -1)
    outlier_mask[initial_outliers] = True
    # Check mean z value for each cluster and label clusters with mean z > tree_z_mean as outliers
    unique_labels = set(labels_xz) | set(labels_yz)  # Union of unique labels from both projections
    for label in unique_labels:
        if label != -1:  # Ignore the outlier label (-1)
            cluster_mask = (labels_xz == label) | (labels_yz == label)
            cluster_mask_indices = np.where(cluster_mask)[0]
            cluster_z_mean = np.mean(points_df["z"][cluster_mask])
            if cluster_z_mean > tree_z_mean:
                outlier_mask[cluster_mask_indices] = True  # Update the outlier mask

    outlier = points_df[outlier_mask]
    ground = points_df[~outlier_mask]
    # # Visualization of init ground
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(outlier['x'], outlier['y'], outlier['z'], label='Outliers')
    # ax.scatter(ground['x'], ground['y'], ground['z'], label='ground')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.legend()
    # plt.show()

    if ground.size <= min_fur_ground:
        return outlier, ground
    else:
        # -------- projection & further separation -------
        init_ground = np.column_stack((ground["x"], ground["y"], ground["z"]))
        pca = PCA(n_components=3)
        pca.fit(init_ground)
        pc1 = pca.components_[0]
        normal_vector = pca.components_[2]
        projected_points = init_ground.dot(np.column_stack((pc1, normal_vector)))

        clustering_norm = DBSCAN(eps=fur_eps, min_samples=fur_min_samples).fit(projected_points)
        labels_norm = clustering_norm.labels_

        fur_outlier_mask = np.zeros(len(ground), dtype=bool)
        fur_outliers = labels_norm == -1
        fur_outlier_mask[fur_outliers] = True

        unique_labels_norm = np.unique(labels_norm)
        for label in unique_labels_norm:
            if label != -1:  # Ignore the outlier label (-1)
                label_indices = np.where(labels_norm == label)[0]
                cluster_z_mean = np.mean(ground["z"][labels_norm == label])
                if cluster_z_mean > tree_z_mean:
                    fur_outlier_mask[label_indices] = True  # Update the outlier mask
        # # Visualization of fur_clusters
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # unique_labels_norm = set(labels_norm)
        # for label in unique_labels_norm:
        #     if label != -1:
        #         ax.scatter(ground['x'][labels_norm == label], ground['y'][labels_norm == label], ground['z'][labels_norm == label], label=f'Cluster norm-{label}')
        # ax.scatter(ground['x'][fur_outlier_mask], ground['y'][fur_outlier_mask], ground['z'][fur_outlier_mask], c='k', label='Outliers')
        # xx, yy = np.meshgrid(range(int(min(ground['x'])), int(max(ground['x']))),
        #                 range(int(min(ground['y'])), int(max(ground['y']))))
        # zz = np.ones_like(xx) * tree_z_mean
        # ax.plot_surface(xx, yy, zz, alpha=0.2, color='r', label='tree_z_mean threshold')
        # ax.set_xlabel('X-project')
        # ax.set_ylabel('Y-project')
        # ax.set_zlabel('Z-project')
        # # ax.legend()
        # plt.show()

        fur_outlier = ground[fur_outlier_mask]
        fur_ground = ground[~fur_outlier_mask]
        all_outliers = np.concatenate([outlier, fur_outlier])
        # # Visualization of fur_ground
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(fur_outlier['x'], fur_outlier['y'], fur_outlier['z'], label='Outliers')
        # ax.scatter(fur_ground['x'], fur_ground['y'], fur_ground['z'], label='ground')
        # ax.set_xlabel('X')
        # ax.set_ylabel('Y')
        # ax.set_zlabel('Z')
        # ax.legend()
        # plt.show()

        return all_outliers, fur_ground


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # -                           ARGPARSE                             -
    # ------------------------------------------------------------------

    # Argparse: Create parser
    parser = argparse.ArgumentParser()
    # Argparse: Add arguments
    parser.add_argument(dest="project_path", help="Path to the project folder, containing Models/Model_i folders")
    parser.add_argument("-slice_interval", "--slice_interval", default=0.4, help="Interval of each slice")
    parser.add_argument(
        "-expected_mean_tolerance", "--expected_mean_tolerance",
        default=0.8,
        help="mean tolerance when comparing mean of fitted gaussian with tree positions from bev.\
                                Should be increased when bev is not quite accurate, esp. predicted bev",
    )
    parser.add_argument("-center_area", "--center_area", default=3, help="radius of center cylinder around tree")
    parser.add_argument(
        "-expected_trunk_radius", "--expected_trunk_radius",
        default=0.2,
        help="average radius of trunks, should be decided based on the dataset",
    )
    # Argparse: Parse arguments
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # -                           SET UP                               -
    # ------------------------------------------------------------------
    slice_interval = float(args.slice_interval)
    center_area_r = float(args.center_area)  
    # parameters setting
    min_points_per_slice = 3
    further_slice_interval = 0.4

    # model path
    MODELS_PATH = Path(f"{args.project_path}/Models")
    IMAGES_LV95_BATCHES = f"{args.project_path}/Images_LV95_batches.csv"
    df_batches = pd.read_csv(IMAGES_LV95_BATCHES)
    max_batch = df_batches["batch"].max()
    nr_batches = int(max_batch) + 1

    # Loop for each model
    for idx in range(nr_batches):
        # parameters setting
        expected_trunk_radius = float(args.expected_trunk_radius) 
        expected_mean_tolerance = float(args.expected_mean_tolerance)
        noise_level = 0.05  # noise level
        expected_trunk_stddev = expected_trunk_radius + noise_level  # Standard deviation in XY plane
        expected_trunk_covariance = np.array(
            [[expected_trunk_stddev**2, 0], [0, expected_trunk_stddev**2]]
        )  # Covariance matrix in XY plane
        covariance_tolerance = 0.15

        # model path
        INPUT_MODEL_PATH = Path(f"{MODELS_PATH}/Model_{idx}/Tree_Segmentation")
        OUTPUT_MODEL_FOLDER = Path(f"{MODELS_PATH}/Model_{idx}/Ground_extraction")
        OUTPUT_MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

        # read tree positions for model_{idx} (in LV95 coord)
        TREE_POSITIONS = Path(f"{MODELS_PATH}/Model_{idx}/tree_positions_shifted_LV95/tree_positions.txt")
        if TREE_POSITIONS.exists():
            tree_df = pd.read_csv(TREE_POSITIONS, delimiter=",", header=0)
            x_tree = tree_df["x"].tolist()
            y_tree = tree_df["y"].tolist()
            z_tree = tree_df["z"].tolist()
        # --------------------------------------------------------------------------------------------
        #                             --- For each tree segment ---
        # --------------------------------------------------------------------------------------------
        trunk_centers = []
        for txt_file in INPUT_MODEL_PATH.glob("*.txt"):
            # ------------------------------------------------------------------------
            #                         --- step1: preprocessing ---
            # ------------------------------------------------------------------------
            # --- Read the points in each tree segement
            points = []
            ground_points = []
            tree_points = []
            with open(txt_file, "r") as file:
                for line in file:
                    if line.startswith("#"):
                        continue
                    parts = line.split()
                    point_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    points.append((point_id, x, y, z))
            points_array = np.array(points, dtype=[("point_id", "i4"), ("x", "f4"), ("y", "f4"), ("z", "f4")])
            # # V - Create a 3D figure
            # fig = plt.figure(figsize=(12, 10))
            # ax = fig.add_subplot(111, projection='3d')
            # ax.set_title(f"{txt_file.stem} : 3D Scatter with Fitted and Expected Gaussians")
            # plt.ion()
            # plt.pause(1) #uncertain
            # # V - Original points in light gray
            # ax.scatter(points_array['x'], points_array['y'], points_array['z'], c='gray', marker='o', alpha=0.3, s=2)

            # --- Define the bounds for the outliers in z direction, (Calculate the Interquartile Range (IQR) for the z-values)
            q1 = np.percentile(points_array["z"], 25)
            q3 = np.percentile(points_array["z"], 75)
            iqr = q3 - q1
            lower_bound = q1 - 1 * iqr  # 1.5
            upper_bound = q3 + 1 * iqr  # 1.5
            filtered_points = points_array[(points_array["z"] >= lower_bound) & (points_array["z"] <= upper_bound)]
            min_z = np.min(filtered_points["z"])
            max_z = np.max(filtered_points["z"])
            ## V - Points after outlier removal in darker gray
            ## ax.scatter(filtered_points['x'], filtered_points['y'], filtered_points['z'], c='gray', marker='o', alpha=0.3, s=3)

            # --- center area segmentation using tree postion from BEV: find tree_idx and Segment an area around the given tree position
            match = re.search(r"\d+", txt_file.stem)
            if match:
                number = int(match.group())
                x_center, y_center, z_center = x_tree[number], y_tree[number], z_tree[number]
                mask = (filtered_points["x"] - x_center) ** 2 + (
                    filtered_points["y"] - y_center
                ) ** 2 <= center_area_r**2
                center_area_df = filtered_points[mask].copy()
                # # V - Points after outlier removal in darker gray
                # ax.scatter(center_area_df['x'], center_area_df['y'], center_area_df['z'], c='white', marker='o', alpha=0.3, s=2)

            # -------------------------------------------------------------------------
            #                         --- step2: slicing & gaussian fitting ---
            # -------------------------------------------------------------------------

            trunk_detected = 0
            top_of_trunk = False
            outlier_points_array = None
            for z in np.arange(min_z, max_z, slice_interval):
                slice_mask = (center_area_df["z"] >= z) & (center_area_df["z"] < z + slice_interval)
                slice_points = center_area_df[slice_mask]
                # # V - Points in the current slice
                ## ax.scatter(slice_points['x'], slice_points['y'], slice_points['z'], c='black', marker='o', s=2)

                # --- preprocessing: If the number of points in the slice is too small, add the points into 'outliers', and continue to the next slice
                if len(slice_points) < min_points_per_slice:
                    if outlier_points_array is None:
                        outlier_points_array = np.array(
                            slice_points, dtype=np.dtype([("point_id", "i4"), ("x", "f4"), ("y", "f4"), ("z", "f4")])
                        )
                    else:
                        new_slice_array = np.array(
                            slice_points, dtype=np.dtype([("point_id", "i4"), ("x", "f4"), ("y", "f4"), ("z", "f4")])
                        )
                        outlier_points_array = np.concatenate((outlier_points_array, new_slice_array))
                    continue

                # --- preprocessing: remove outliers according to density in each slice
                candidate_trunk, candidate_ground = filter_by_density(
                    [x_center, y_center], slice_points, eps=0.5, min_samples=6
                )
                if candidate_trunk is not None:
                    filtered_points = candidate_trunk
                else:
                    filtered_points = slice_points
                slice_points_for_gmm = np.vstack((filtered_points["x"], filtered_points["y"], filtered_points["z"])).T

                # --- Apply Gaussian Mixture Model to the slice
                gmm = GaussianMixture(n_components=1, covariance_type="full").fit(slice_points_for_gmm)
                # # V -  3d gaussian vidualization
                # plot3d_elliptical_cylinder(ax, gmm.means_[0], gmm.covariances_[0], slice_interval, color='red', alpha=0.5)
                # plot3d_elliptical_cylinder(ax, gmm.means_[0], expected_trunk_covariance, slice_interval, color='blue', alpha=0.5)
                # # Pause for a short while to see the plot
                # plt.draw()
                # plt.pause(1)
                # plt.show()

                # ---- trunk detection
                # covariance and mean of fitted gaussian vs expected covariance and mean for the trunk,
                mean_from_center = np.linalg.norm(gmm.means_[0][:2] - np.array([x_center, y_center]))
                eigenvalues, _ = np.linalg.eigh(gmm.covariances_[0][:2, :2])
                lambda1, lambda2 = eigenvalues
                R = np.sqrt(lambda1 * lambda2)
                canopy_points_array = np.array([])

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
                            trunk_points, dtype=np.dtype([("point_id", "i4"), ("x", "f4"), ("y", "f4"), ("z", "f4")])
                        )
                    else:
                        new_points_array = np.array(
                            trunk_points, dtype=np.dtype([("point_id", "i4"), ("x", "f4"), ("y", "f4"), ("z", "f4")])
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
                            canopy_points, dtype=np.dtype([("point_id", "i4"), ("x", "f4"), ("y", "f4"), ("z", "f4")])
                        )
                        print(f"{txt_file.stem} canopy{z} detected successfully")
                        break  # Exit the loop once the trunk is found

            if trunk_detected == 0:
                print(f"{txt_file.stem} detected failed !!!")
                continue
            left_points_mask = points_array["z"] < z
            left_points = points_array[left_points_mask]
            left_points_array = np.array(
                left_points, dtype=np.dtype([("point_id", "i4"), ("x", "f4"), ("y", "f4"), ("z", "f4")])
            )

            if outlier_points_array is not None:
                outlier_ids = outlier_points_array["point_id"]
                mask = ~np.isin(left_points["point_id"], outlier_ids)
                left_points = left_points[mask]

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
            trunk_z_mean = np.mean(fur_trunk["z"])
            trunk_id = fur_trunk["point_id"]
            mask = ~np.isin(starting_points["point_id"], trunk_id)
            left_wo_trunk = starting_points[mask]
            # # VV -
            # ax.scatter(left_wo_trunk['x'], left_wo_trunk['y'], left_wo_trunk['z'], c='white', marker='o', alpha=0.3, s=2)

            fur_eps = 0.3
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

            trunk_id = fur_trunk["point_id"]
            ground_id = ground_points["point_id"]
            mask = ~np.isin(points_array["point_id"], np.concatenate([trunk_id, ground_id]))
            else_points = points_array[mask]

            # ------------------------------ initial saving ---------------------------------
            if canopy_points_array.size > 0:
                canopy_points_to_save = np.column_stack(
                    (
                        canopy_points_array["point_id"],
                        canopy_points_array["x"],
                        canopy_points_array["y"],
                        canopy_points_array["z"],
                    )
                )
            left_points_to_save = np.column_stack(
                (left_points_array["point_id"], left_points_array["x"], left_points_array["y"], left_points_array["z"])
            )
            trunk_points_to_save = np.column_stack(
                (
                    trunk_points_array["point_id"],
                    trunk_points_array["x"],
                    trunk_points_array["y"],
                    trunk_points_array["z"],
                )
            )
            # Save the tree and ground points to separate files
            # Define the paths for the output files
            init_tree_file_folder = Path(f"{OUTPUT_MODEL_FOLDER}/init_saperation")
            init_tree_file_folder.mkdir(parents=True, exist_ok=True)
            init_canopy_file_folder = Path(f"{init_tree_file_folder}/canopy")
            init_canopy_file_folder.mkdir(parents=True, exist_ok=True)
            left_points_file_folder = Path(f"{init_tree_file_folder}/left_points")
            left_points_file_folder.mkdir(parents=True, exist_ok=True)
            trunk_points_file_folder = Path(f"{init_tree_file_folder}/trunk_points")
            trunk_points_file_folder.mkdir(parents=True, exist_ok=True)

            init_canopy_file_path = init_canopy_file_folder / f"init_canopy_points_{txt_file.stem}.txt"
            left_points_file_path = left_points_file_folder / f"init_left_points_{txt_file.stem}.txt"
            trunk_points_file_path = trunk_points_file_folder / f"init_trunk_points_{txt_file.stem}.txt"

            # Save the points to the output files with the appropriate headers
            if canopy_points_array.size > 0:
                np.savetxt(
                    init_canopy_file_path,
                    canopy_points_to_save,
                    fmt="%d %f %f %f",
                    header=f"# 3D point list with one line of data per point:\n"
                    f"#   POINT3D_ID, X, Y, Z\n"
                    f"# Number of points: {len(canopy_points_to_save)}",
                    comments="",
                )

            np.savetxt(
                left_points_file_path,
                left_points_to_save,
                fmt="%d %f %f %f",
                header=f"# 3D point list with one line of data per point:\n"
                f"#   POINT3D_ID, X, Y, Z\n"
                f"# Number of points: {len(left_points_to_save)}",
                comments="",
            )

            np.savetxt(
                trunk_points_file_path,
                trunk_points_to_save,
                fmt="%d %f %f %f",
                header=f"# 3D point list with one line of data per point:\n"
                f"#   POINT3D_ID, X, Y, Z\n"
                f"# Number of points: {len(trunk_points_to_save)}",
                comments="",
            )
            print(f"initial separation for tree{txt_file} of model{idx} completed")

            # -----------------------------further save---------------------------------------
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
                fmt="%d %f %f %f",
                header=f"# 3D point list with one line of data per point:\n"
                f"#   POINT3D_ID, X, Y, Z\n"
                f"# Number of points: {len(fur_trunk)}",
                comments="",
            )

            np.savetxt(
                ground_points_file_path,
                ground_points,
                fmt="%d %f %f %f",
                header=f"# 3D point list with one line of data per point:\n"
                f"#   POINT3D_ID, X, Y, Z\n"
                f"# Number of points: {len(ground_points)}",
                comments="",
            )

            np.savetxt(
                else_points_file_path,
                else_points,
                fmt="%d %f %f %f",
                header=f"# 3D point list with one line of data per point:\n"
                f"#   POINT3D_ID, X, Y, Z\n"
                f"# Number of points: {len(else_points)}",
                comments="",
            )
            print(f"further separation for tree{txt_file} of model{idx} completed")

        # --------------------------------------------------------------------------------------
        #                         --- For every entire model ---
        #    integrate to 3 parts: final ground & trunk & else(canopy & outliers)
        # --------------------------------------------------------------------------------------
        trunk_center_path = OUTPUT_MODEL_FOLDER / f"trunk_centers.txt"
        with open(trunk_center_path, "w") as file:
            file.write("tree_idx x y z\n")
            for center in trunk_centers:
                file.write(f"{int(center[0])} {center[1]:.2f} {center[2]:.2f} {center[3]:.2f}\n")

        point_types_info = {
            "trunk": {
                "pattern": "trunk_tree_*_segmented_area.txt",
                "read_file": fur_trunk_file_folder,
                "output_file": f"trunk_model{idx}.txt",
            },
            "ground": {
                "pattern": "ground_tree_*_segmented_area.txt",
                "read_file": ground_points_file_folder,
                "output_file": f"seg_ground_model{idx}.txt",
            },
            "else_points": {
                "pattern": "else_tree_*_segmented_area.txt",
                "read_file": else_points_file_folder,
                "output_file": f"seg_else_model{idx}.txt",
            },
        }
        # ---

        for point_type, info in point_types_info.items():
            pattern = info["pattern"]
            file_list = glob.glob(os.path.join(info["read_file"], pattern))

            all_content = []
            total_points = 0
            for file_path in file_list:
                with open(file_path, "r") as file:
                    content = file.readlines()
                    all_content.extend(content[3:])
                    total_points += int(content[2].split(":")[1].strip())

            # ---------------- ground refine -----------------
            if point_type == "ground":
                # Assuming all_content is already populated as before
                pointIDs = []
                X_values = []
                Y_values = []
                Z_values = []

                for line in all_content:
                    values = line.split()
                    pointIDs.append(int(values[0]))
                    X_values.append(float(values[1]))
                    Y_values.append(float(values[2]))
                    Z_values.append(float(values[3]))

                entire_ground = {
                    "id": np.array(pointIDs),
                    "x": np.array(X_values),
                    "y": np.array(Y_values),
                    "z": np.array(Z_values),
                }

                Eground = np.column_stack((entire_ground["x"], entire_ground["y"], entire_ground["z"]))
                pca = PCA(n_components=3)
                pca.fit(Eground)
                pc1 = pca.components_[0]
                normal_vector = pca.components_[2]

                projected_ground_3d = np.dot(Eground, normal_vector)
                median_height_all = np.median(projected_ground_3d)
                ground_height_outlier_filter = 0.8  # 0.8
                Eoutlier_mask = projected_ground_3d > median_height_all + ground_height_outlier_filter

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
                    fmt="%d %f %f %f",
                    header=f"# 3D point list with one line of data per point:\n"
                    f"#   POINT3D_ID, X, Y, Z\n"
                    f"# Number of points: {len(fin_ground)}",
                    comments="",
                )

            # ------------------- end --------------------------
            integrate_file_path = os.path.join(OUTPUT_MODEL_FOLDER, info["output_file"])
            with open(integrate_file_path, "w") as output_file:
                # Write the updated header to the output file
                output_file.write("# 3D point list with one line of data per point:\n")
                output_file.write("#   POINT3D_ID, X, Y, Z\n")
                output_file.write(f"# Number of points: {total_points}\n")

                # Write the integrated content to the output file
                output_file.writelines(all_content)

        print(f"model{idx} completed")
    print("Done!")
