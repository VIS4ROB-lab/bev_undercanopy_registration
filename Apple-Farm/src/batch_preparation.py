import pycolmap
import numpy as np
import pandas as pd
from pathlib import Path
from subprocess import call
import argparse
import math
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import cv2
import numpy as np
import networkx as nx
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
import matplotlib.patches as patches
import shutil

from colmap_vidual import colmap_to_txt


def rotate_points(points, angle):
    """Rotates points counterclockwise by a given angle."""
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.dot(points, rotation_matrix)


def visualize_hull_and_long_edge(points, long_edge_pair):
    plt.scatter(points[:, 0], points[:, 1], label="Points")
    hull = ConvexHull(points)
    for simplex in hull.simplices:
        plt.plot(points[simplex, 0], points[simplex, 1], "k-")

    # Highlight the longest edge
    plt.plot(*zip(*long_edge_pair), "r-", linewidth=2, label="Longest Edge")

    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.title("Convex Hull and Longest Side of Bounding Box")
    plt.legend()
    plt.show()


def find_longest_edge(points):
    hull = ConvexHull(points)
    hull_points = points[hull.vertices]
    hull_area = hull.volume

    min_area_difference = float("inf")
    for i in range(len(hull_points)):
        p1 = hull_points[i]
        p2 = hull_points[(i + 1) % len(hull_points)]

        # Calculate angle to horizontal
        angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
        # Rotate all points
        rotated_points = rotate_points(points, angle)
        # Compute the bounding box
        min_x, max_x = np.min(rotated_points[:, 0]), np.max(rotated_points[:, 0])
        min_y, max_y = np.min(rotated_points[:, 1]), np.max(rotated_points[:, 1])
        width, height = max_x - min_x, max_y - min_y
        bounding_box_area = width * height

        area_difference = abs(bounding_box_area - hull_area)
        if area_difference < min_area_difference:
            rot_points = rotated_points
            min_area_difference = area_difference
            max_pair = np.array([p1, p2])

    visualize_hull_and_long_edge(rot_points, max_pair)
    visualize_hull_and_long_edge(points, max_pair)
    return max_pair


def calculate_homogeneous_rotation_matrix(edge):
    p1, p2 = edge
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    # ensure acute angle
    if angle < -np.pi / 2:
        angle += np.pi
    elif angle > np.pi / 2:
        angle -= np.pi
    cos_a, sin_a = np.cos(-angle), np.sin(-angle)
    rotation_matrix = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]])
    # no translation is needed
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    return transformation_matrix


def find_tree_pairs(points_source, points_target, max_dist):
    corresponding_tree_indexes = []
    for target_id, point_target in enumerate(points_target):
        source_id = None
        min_found_dist = math.inf
        for i_source, point_source in enumerate(points_source):
            dist = np.linalg.norm(point_target - point_source)
            if dist <= max_dist and dist < min_found_dist:
                min_found_dist = dist
                source_id = i_source
        if source_id is not None:
            corresponding_tree_indexes.append((source_id, target_id))
    return corresponding_tree_indexes


def calculate_error_similarity(error_vector1, error_vector2, mag_weight=0.5, angle_weight=0.5):
    # Calculate the magnitude difference using Euclidean distance
    magnitude_diff = np.linalg.norm(error_vector1 - error_vector2)

    # the directional difference
    unit_vector1 = error_vector1 / np.linalg.norm(error_vector1)
    unit_vector2 = error_vector2 / np.linalg.norm(error_vector2)
    cos_angle = np.dot(unit_vector1, unit_vector2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # cosine value [-1,1]
    angle_diff = np.arccos(cos_angle)  # angular difference in radians

    # combine
    error_similarity = np.exp(-magnitude_diff) * (1 - angle_diff / np.pi)
    # error_similarity = (mag_weight * magnitude_similarity) + (angle_weight * angular_similarity)
    return error_similarity


def save_points_to_file(filename, points, indices, header="x,y,z,tree_name"):
    data_to_save = np.hstack((points, indices.reshape(-1, 1)))  # Combine points with their indices
    np.savetxt(
        filename, data_to_save, delimiter=",", header=header, fmt=["%f", "%f", "%f", "%d"], comments=""
    )  # fmt='%s' ; fmt=['%f', '%f', '%f', '%d']


def plot_bounding_box(bounding_boxes_array):
    plt.figure(figsize=(10, 8))

    for bbox in bounding_boxes_array:
        cluster, xmin, xmax, ymin, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin
        # print(f"Cluster: {cluster}, xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}, width: {width}, height: {height}")

        # Create a rectangle patch for each bounding box
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=1, edgecolor="r", facecolor="none")

        plt.gca().add_patch(rect)
        plt.text(xmin, ymin, f"Cluster {int(cluster)}", fontsize=8, verticalalignment="bottom")

    plt.xlim([xmin - 10, xmax + 10])
    plt.ylim([ymin - 10, ymax + 10])
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Bounding Visualization")
    plt.show()


def transform_points(points_with_index, trans_matrix):
    homogeneous_points = np.hstack((points_with_index[:, :3], np.ones((points_with_index.shape[0], 1))))
    transformed_points = homogeneous_points.dot(trans_matrix.T)
    return np.hstack((transformed_points[:, :3], points_with_index[:, 3:]))


if __name__ == "__main__":

    # Argparse: Create parser
    parser = argparse.ArgumentParser()
    # Argparse: Add arguments
    parser.add_argument(
        dest="project_path", help="Path to the project folder, containing Models/Model_i folders with sparse models"
    )
    parser.add_argument("-similarity_threshold", default=0.6)
    parser.add_argument("-tree_pair_distance", default=2.0)

    # Argparse: Parse arguments
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # -                           Set up                             -
    # ------------------------------------------------------------------
    PROJECT_PATH = f"{args.project_path}"

    # Read number of batches/models
    IMAGES_LV95_BATCHES = f"{args.project_path}/Images_LV95_batches.csv"
    df_batches = pd.read_csv(IMAGES_LV95_BATCHES)
    max_batch = df_batches["batch"].max()
    nr_batches = int(max_batch) + 1
    print("nr of batches:", nr_batches)

    # ---------------------------------------------------------------------------------
    # -----     transform models ((aligned to GT) initial aligned -> aligned )    -----
    # ---------------------------------------------------------------------------------
    MODEL_DIR = Path(f"{args.project_path}/Models")
    EVALUATION_DIR = Path(f"{args.project_path}/Correctness_loop/evaluation")
    INIT_ICP_PATH = Path(f"{args.project_path}/Correctness_loop/initial_alignment")

    for idx in range(1, nr_batches):
        init_model_path = Path(f"{INIT_ICP_PATH}/Output/Init_transed_models/model_{idx}")
        init_model = pycolmap.Reconstruction(init_model_path)

        align_tran_file = f"{EVALUATION_DIR}/Transformation/final_transformation_m{idx}.txt"
        align_tran = np.loadtxt(align_tran_file)

        init_model.transform(pycolmap.SimilarityTransform3(align_tran[:3]))
        OUTPUT_TRANS_MODEL_PATH = Path(f"{EVALUATION_DIR}/aligned_model/model{idx}")
        OUTPUT_TRANS_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        init_model.write(OUTPUT_TRANS_MODEL_PATH)
        print("apply the alignment transformation matrix successfully")

        # VIDUALIZATION
        # Convert model to txt format
        TXT_PATH = Path(f"{EVALUATION_DIR}/aligned_model/model{idx}_txt")
        TXT_PATH.mkdir(parents=True, exist_ok=True)
        CAM_SAVE_FILE = f"{TXT_PATH}/cam_loop_eval_m{idx}.txt"

        colmap_to_txt(OUTPUT_TRANS_MODEL_PATH, TXT_PATH, CAM_SAVE_FILE)

        shutil.copy(CAM_SAVE_FILE, f"{EVALUATION_DIR}/Camera_poses_compare/cam_eval_m{idx}_loop.txt")

    # ---------------------------------------------------------------
    # ------        transform models (align axes )              -----
    # ---------------------------------------------------------------
    INPUT_V = Path(f"{PROJECT_PATH}/Correctness_loop/1_ICP_models/Output")
    INPUT_V_TREE = Path(f"{PROJECT_PATH}/Correctness_loop/1_ICP_models/Input")
    INPUT_H = Path(f"{PROJECT_PATH}/Correctness_loop/2_horizontal/Output")
    INPUT_MODEL = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/aligned_model")
    INPUT_BASE_MODEL = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/gt_models")

    BATCH_ALIGN_FOLDER = Path(f"{PROJECT_PATH}/Correctness_loop/3_batch_align")
    BATCH_ALIGN_FOLDER.mkdir(parents=True, exist_ok=True)
    BATCH_INIT_PATH = Path(f"{BATCH_ALIGN_FOLDER}/Init_files")
    BATCH_INIT_PATH.mkdir(parents=True, exist_ok=True)
    BATCH_INIT_MODEL = Path(f"{BATCH_INIT_PATH}/Models")
    BATCH_INIT_MODEL.mkdir(parents=True, exist_ok=True)
    BATCH_INIT_GROUND = Path(f"{BATCH_INIT_PATH}/Grounds")
    BATCH_INIT_GROUND.mkdir(parents=True, exist_ok=True)
    BATCH_INIT_TREES = Path(f"{BATCH_INIT_PATH}/Tree_centers")
    BATCH_INIT_TREES.mkdir(parents=True, exist_ok=True)
    TRANS_FILE = f"{BATCH_INIT_PATH}/trans_m1_xy.txt"

    # -------- calculate the transformation matrix ----------
    for idx in range(1, 2):
        TREE_POSITION_FILE = (
            f"{PROJECT_PATH}/Correctness_loop/2_horizontal/Output/Tree_centers/transformed_tree_pos_model_1.txt"
        )
        data = np.genfromtxt(TREE_POSITION_FILE, dtype=None, skip_header=1, names=["tree_idx", "x", "y", "z"])
        tree_indices = data["tree_idx"]
        tree_positions = np.vstack((data["x"], data["y"], data["z"])).T

        longest_edge = find_longest_edge(tree_positions[:, :2])
        transformation_matrix = calculate_homogeneous_rotation_matrix(longest_edge)
        # save transformation
        TRANS_FILE = f"{BATCH_INIT_PATH}/trans_m1_xy.txt"
        np.savetxt(TRANS_FILE, transformation_matrix)
        print("transformation for aligning to axes successfully calculated")

    for idx in range(nr_batches):
        # transformation_matrix = np.loadtxt(TRANS_FILE)
        # Transform tree positions
        if idx == 0:
            pc_tree_pos = o3d.io.read_point_cloud(f"{INPUT_V_TREE}/init_transed_tree_centers_m0.ply")
            INPUT_TREE_TXT = f"{INPUT_V_TREE}/init_transed_tree_centers_m0.txt"
        else:
            pc_tree_pos = o3d.io.read_point_cloud(f"{INPUT_H}/Tree_centers/transformed_tree_pos_model_{idx}.ply")
            INPUT_TREE_TXT = f"{INPUT_H}/Tree_centers/transformed_tree_pos_model_{idx}.txt"
        pc_tree_pos_transformed = pc_tree_pos.transform(transformation_matrix)
        output_tree_pos_path = f"{BATCH_INIT_TREES}/transformed_tree_pos_model_{idx}.ply"
        o3d.io.write_point_cloud(output_tree_pos_path, pc_tree_pos_transformed)

        tree_df = pd.read_csv(INPUT_TREE_TXT, delimiter=r"\s+", header=0)
        idx_tree = tree_df["tree_idx"].tolist()
        transformed_coordinates = np.asarray(pc_tree_pos_transformed.points)
        with open(f"{BATCH_INIT_TREES}/transformed_tree_pos_model_{idx}.txt", "w") as f:
            f.write("tree_idx x y z\n")
            for index, coords in zip(idx_tree, transformed_coordinates):
                f.write(f"{index} {coords[0]:.2f} {coords[1]:.2f} {coords[2]:.2f}\n")

        # Transform Ground
        if idx == 0:
            GROUND_FILE = f"{INPUT_V}/ground_models/fixed_ground_model_0.ply"
        else:
            GROUND_FILE = f"{INPUT_H}/ground_models/transformed_ground_model_{idx}.ply"
        ground = o3d.io.read_point_cloud(f"{GROUND_FILE}")
        transformed_ground = ground.transform(transformation_matrix)
        o3d.io.write_point_cloud(f"{BATCH_INIT_GROUND}/transformed_ground_model_{idx}.ply", transformed_ground)
        # .txt
        points_array = np.asarray(transformed_ground.points)
        colors_array = np.asarray(transformed_ground.colors)
        transformed_source_data = {
            "x": points_array[:, 0],
            "y": points_array[:, 1],
            "z": points_array[:, 2],
            "r": colors_array[:, 0] * 255,
            "g": colors_array[:, 1] * 255,
            "b": colors_array[:, 2] * 255,
        }
        transformed_source_df = pd.DataFrame(transformed_source_data)
        transformed_source_txt_path = f"{BATCH_INIT_GROUND}/transformed_ground_model_{idx}.txt"
        transformed_source_df.to_csv(transformed_source_txt_path, index=False, sep=" ", header=True)

        # Transform models
        if idx == 0:
            MODEL_PATH = f"{INPUT_BASE_MODEL}/GT_Model_0"
        else:
            MODEL_PATH = f"{INPUT_MODEL}/model{idx}"
        model = pycolmap.Reconstruction(MODEL_PATH)
        model.transform(pycolmap.SimilarityTransform3(transformation_matrix[:3]))
        OUTPUT_TRANS_MODEL_PATH = Path(f"{BATCH_INIT_MODEL}/model{idx}")
        OUTPUT_TRANS_MODEL_PATH.mkdir(parents=True, exist_ok=True)
        model.write(OUTPUT_TRANS_MODEL_PATH)
        # TXT
        TXT_PATH = Path(f"{BATCH_INIT_MODEL}/model{idx}_txt")
        TXT_PATH.mkdir(parents=True, exist_ok=True)
        cam_pose_file = f"{TXT_PATH}/batch_init_cam_m{idx}.txt"
        colmap_to_txt(OUTPUT_TRANS_MODEL_PATH, TXT_PATH, cam_pose_file)

        print(f"all in model{idx} aligned to axes")

    # -----------  bounding boxes for each area --------------
    TREE_CENTERS_EVAL_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval")

    BATCH_AREAS_PATH = Path(f"{BATCH_ALIGN_FOLDER}/Batches")
    BATCH_AREAS_PATH.mkdir(parents=True, exist_ok=True)

    TRANS_FILE = f"{BATCH_INIT_PATH}/trans_m1_xy.txt"
    axes_align_trans_matrix = np.loadtxt(TRANS_FILE)
    inverse_trans_matrix = np.linalg.inv(axes_align_trans_matrix)

    TREE_POSITION_BASE_FILE = f"{BATCH_INIT_TREES}/transformed_tree_pos_model_0.txt"
    data_base = np.genfromtxt(TREE_POSITION_BASE_FILE, dtype=None, skip_header=1, names=["tree_idx", "x", "y", "z"])
    tree_indices_desire = data_base["tree_idx"]
    tree_positions_desire = np.vstack((data_base["x"], data_base["y"], data_base["z"])).T

    for idx in range(1, nr_batches):
        CROPPED_TRESS_PATH = Path(f"{BATCH_AREAS_PATH}/Cropped_trees{idx}")
        CROPPED_TRESS_PATH.mkdir(parents=True, exist_ok=True)

        # input tree centers
        TREE_POSITION_FILE = f"{BATCH_INIT_TREES}/transformed_tree_pos_model_{idx}.txt"
        data = np.genfromtxt(TREE_POSITION_FILE, dtype=None, skip_header=1, names=["tree_idx", "x", "y", "z"])
        tree_indices = data["tree_idx"]
        tree_positions = np.vstack((data["x"], data["y"], data["z"])).T

        # Find tree pairs
        max_dist = float(args.tree_pair_distance)
        tree_pairs = find_tree_pairs(tree_positions, tree_positions_desire, max_dist)

        start_points_index = tree_indices[[pair[0] for pair in tree_pairs]]
        end_points_index = tree_indices_desire[[pair[1] for pair in tree_pairs]]
        indices_combined = np.vstack((end_points_index, start_points_index)).T
        np.savetxt(
            f"{BATCH_INIT_TREES}/tree_pairs_m{idx}.txt",
            indices_combined,
            fmt="%d",
            delimiter=",",
            header=f"Index_base,Index_model",
            comments="",
        )
        shutil.copy(f"{BATCH_INIT_TREES}/tree_pairs_m{idx}.txt", f"{TREE_CENTERS_EVAL_PATH}/tree_pairs_m{idx}.txt")

        # Calculate error vectors
        error_vectors = np.array(
            [tree_positions_desire[pair[1], :2] - tree_positions[pair[0], :2] for pair in tree_pairs]
        )

        # vidualize tree pairs & error vectors
        start_points = tree_positions[[pair[0] for pair in tree_pairs]]
        end_points = tree_positions_desire[[pair[1] for pair in tree_pairs]]
        plt.figure(figsize=(10, 10.5))
        plt.scatter(
            start_points[:, 0], start_points[:, 1], color="red", label="Model to be aligned", s=150
        )  # Current Position
        plt.scatter(end_points[:, 0], end_points[:, 1], color="green", label="Base Model", s=150)  # Desired Position
        plt.quiver(
            start_points[:, 0],
            start_points[:, 1],
            error_vectors[:, 0],
            error_vectors[:, 1],
            angles="xy",
            scale_units="xy",
            scale=0.05,
            color="blue",
            alpha=0.5,
            label="Error Vectors",
        )
        plt.legend(fontsize=22, loc="lower left")
        plt.grid(True)
        plt.title("Visualization of Tree Position Errors", fontsize=28)
        plt.xlabel("X Coordinate (m)", fontsize=26)
        plt.ylabel("Y Coordinate (m)", fontsize=26)
        xmin, xmax = min(start_points[:, 0]), max(start_points[:, 0])
        ymin, ymax = min(start_points[:, 1]), max(start_points[:, 1])
        plt.xlim(xmin - 10, xmax + 5)
        plt.ylim(ymin - 5, ymax + 10)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig(f"{BATCH_AREAS_PATH}/error_vectors_model{idx}.png")
        plt.show()
        plt.close()

        # ----- Combine (x, y) coordinates and error vectors into a single feature set
        tree_pairs_array = np.array(tree_pairs)

        # features: error polar
        lengths = np.linalg.norm(error_vectors, axis=1)  # Length of each vector
        angles = np.arctan2(error_vectors[:, 1], error_vectors[:, 0])  # Angle with respect to x-axis
        error_polar = np.column_stack((lengths, angles))

        points = tree_positions[tree_pairs_array[:, 0], :2]  # 2d
        points_3d = tree_positions[tree_pairs_array[:, 0]]

        # ----- graph ----
        # Initialize Subdiv2D with the bounding rectangle
        rect = cv2.boundingRect(points.astype(np.float32))
        subdiv = cv2.Subdiv2D(rect)
        for p in points:
            subdiv.insert((p[0], p[1]))
        edges = subdiv.getEdgeList()
        # Construct the graph using the edge list
        G = nx.Graph()
        G_prune = nx.Graph()
        similarity_threshold = float(args.similarity_threshold)  # 0.6 case2; 0.7 case 3
        for edge in edges:
            pt1 = (edge[0], edge[1])
            pt2 = (edge[2], edge[3])

            # if abs(pt1[0]) > 45 or abs(pt1[1]) > 40 or abs(pt2[0]) > 45 or abs(pt2[1]) > 40:
            #     continue
            # Use np.isclose to handle floating-point precision issues
            idx1, idx2 = -1, -1
            for i, p in enumerate(points):
                if np.isclose(p, pt1).all():
                    idx1 = i
                if np.isclose(p, pt2).all():
                    idx2 = i

            # Add an edge between the corresponding points' indices, with weight!
            if idx1 >= 0 and idx2 >= 0:
                error_similarity = calculate_error_similarity(error_vectors[idx1], error_vectors[idx2])
                G.add_edge(idx1, idx2, weight=error_similarity)
                G.add_node(idx1, pos=points[idx1])
                G.add_node(idx2, pos=points[idx2])

                if error_similarity >= similarity_threshold:
                    G_prune.add_edge(idx1, idx2, weight=error_similarity)  # , weight=error_similarity
                    G_prune.add_node(idx1, pos=points[idx1])
                    G_prune.add_node(idx2, pos=points[idx2])

        # vidualization
        pos = {i: points[i] for i in range(len(points))}

        plt.figure(figsize=(10, 10))
        # nx.draw(G, pos, node_size=50, with_labels=True)
        nx.draw_networkx(G_prune, pos, node_size=200, with_labels=True)
        plt.title("Graph Visualization of Tree Centers")
        plt.legend()
        plt.xlabel(
            "X Coordinate (m)",
        )
        plt.ylabel("Y Coordinate (m)")
        plt.grid(True)
        # plt.savefig(f'{BATCH_AREAS_PATH}/graph_visualization.png') #G
        plt.savefig(f"{BATCH_AREAS_PATH}/graph_connection_m{idx}.png")  # G_prune
        plt.show()
        plt.close()

        # ---- cluster in graph ---
        # -- edge pruning based on a threshold + scipy.sparse.csgraph.connected_components
        node_mapping = {node: i for i, node in enumerate(G_prune.nodes())}
        mapped_G_prune = nx.relabel_nodes(G_prune, node_mapping)
        nx_sparse_array = nx.to_scipy_sparse_array(
            mapped_G_prune
        )  # , nodelist=sorted(G_prune.nodes()), weight='weight'
        # print(nx_sparse_array.toarray())
        # print(nx_sparse_array)
        n_components, labels = connected_components(csgraph=nx_sparse_array, directed=False, return_labels=True)
        # print(labels)
        # vidual
        colors = plt.cm.jet(np.linspace(0, 1, n_components))
        plt.figure(figsize=(10, 10))

        # Initialize an array to store the cluster label for each tree position
        cluster_labels = np.full(
            tree_positions[tree_pairs_array[:, 0]].shape[0], -1
        )  # Initialize with -1 to indicate no cluster

        for component in range(n_components):
            node_indices = np.where(labels == component)[0]
            subgraph = G_prune.subgraph(node_indices)

            # node_positions = np.array([points[node_index] for node_index in node_indices])
            original_indices = [list(node_mapping.keys())[list(node_mapping.values()).index(i)] for i in node_indices]
            node_positions = np.array([points[node_index] for node_index in original_indices])
            plt.scatter(
                node_positions[:, 0], node_positions[:, 1], s=300, c=[colors[component]], label=f"Cluster {component}"
            )

            # Update the cluster_labels array
            for i in original_indices:
                cluster_labels[i] = component

        # save cluster info
        graph_clusters_str = np.array([str(cluster) for cluster in cluster_labels])[:, np.newaxis]

        indices = np.arange(graph_clusters_str.shape[0]).reshape(-1, 1)
        tree_positions_with_clusters = np.hstack(
            (
                tree_indices[tree_pairs_array[:, 0]][:, np.newaxis],
                tree_positions[tree_pairs_array[:, 0]],
                graph_clusters_str,
            )
        )

        header = "tree_idx x y z cluster_label"
        np.savetxt(
            f"{BATCH_AREAS_PATH}/clustered_graph_m{idx}.csv",
            tree_positions_with_clusters,
            delimiter=" ",
            header=header,
            fmt="%s",
            comments="",
        )

        # cluster vidual
        plt.title("Clusters of Tree Centers with Connectivity", fontsize=28)
        plt.xlabel("X Coordinate (m)", fontsize=26)
        plt.ylabel("Y Coordinate (m)", fontsize=26)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend()
        plt.savefig(f"{BATCH_AREAS_PATH}/cluster_graph_m{idx}.png")
        # plt.show()
        plt.close()

        # # -- seeded kmeans --
        unique_clusters = np.unique(graph_clusters_str)
        centroids = np.array(
            [
                points[graph_clusters_str.flatten() == cluster].mean(axis=0)
                for cluster in unique_clusters
                if cluster != "-1"
            ]
        )

        kmeans = KMeans(n_clusters=n_components, init=centroids, n_init=1)
        kmeans.fit(points)
        updated_centroids = kmeans.cluster_centers_
        kmeans_labels = kmeans.labels_

        final_clusters_str = np.array([str(cluster) for cluster in kmeans_labels])[:, np.newaxis]
        tree_positions_with_clusters = np.hstack(
            (
                tree_indices[tree_pairs_array[:, 0]][:, np.newaxis],
                tree_positions[tree_pairs_array[:, 0]],
                final_clusters_str,
            )
        )
        header = "tree_idx x y z cluster_label"
        np.savetxt(
            f"{BATCH_AREAS_PATH}/clustered_graph_kmeans_m{idx}.csv",
            tree_positions_with_clusters,
            delimiter=" ",
            header=header,
            fmt="%s",
            comments="",
        )

        plt.figure(figsize=(10, 10.5))
        tree_positions_with_clusters_2d = np.hstack((points, final_clusters_str))
        areas = []
        for i in np.unique(kmeans_labels):
            cluster_points = points[kmeans_labels == i]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}", s=260)

            # Update the bouding box
            padding = 5  # 3 #13
            xmin = cluster_points[:, 0].min() - padding
            xmax = cluster_points[:, 0].max() + padding
            ymin = cluster_points[:, 1].min() - padding
            ymax = cluster_points[:, 1].max() + padding
            areas.append([xmin, xmax, ymin, ymax])

        plt.xlabel("X Coordinate (m)", fontsize=26)
        plt.ylabel("Y Coordinate (m)", fontsize=26)
        # plt.xlim([-51,22])
        # plt.ylim([-46,10])
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.title("Clustered Trees (model to be aligned)", fontsize=28)
        plt.legend(fontsize=22, loc="lower left")
        plt.savefig(f"{BATCH_AREAS_PATH}/cluster_graph_kmeans_m{idx}.png")
        plt.show()

        # # bounding box vidual
        # bounding_boxes_array = np.array(bounding_box)
        # plot_bounding_box(bounding_boxes_array)

        # save bounding box
        output_file_path = f"{BATCH_AREAS_PATH}/bounding_boxes_m{idx}.txt"
        with open(output_file_path, "w") as file:
            file.write("x_start x_end y_start y_end\n")
            for area in areas:
                file.write(f"{area[0]} {area[1]} {area[2]} {area[3]}\n")
        print("bounding boxes calculated")

    # ---------------------------------------------------------------
    # ------                      Cropper                       -----
    # ---------------------------------------------------------------
    for idx in range(1, nr_batches):
        INPUT_MODEL_PATH = Path(f"{BATCH_INIT_MODEL}/model{idx}")
        CROPPED_GROUND_PATH = Path(f"{BATCH_ALIGN_FOLDER}/Batches/Cropped_ground{idx}")
        CROPPED_GROUND_PATH.mkdir(parents=True, exist_ok=True)
        CROPPED_MODEL_PATH = Path(f"{BATCH_ALIGN_FOLDER}/Batches/Cropped_model{idx}")
        CROPPED_MODEL_PATH.mkdir(parents=True, exist_ok=True)

        # Read boundaries from file
        boundary_file_path = f"{BATCH_AREAS_PATH}/bounding_boxes_m{idx}.txt"
        with open(boundary_file_path, "r") as file:
            next(file)  # Skip header line
            boundaries = [line.strip().split() for line in file]

        # Iterate over the boundaries and run the cropper for each
        # ground
        for i, boundary in enumerate(boundaries):
            segment_min_x, segment_max_x, segment_min_y, segment_max_y = map(float, boundary)

            ground_file = Path(f"{BATCH_INIT_GROUND}/transformed_ground_model_{idx}.txt")
            with open(ground_file, "r") as file:
                header = next(file)
                filtered_points = []
                for line in file:
                    x, y, z, *_ = map(float, line.split())
                    if segment_min_x <= x <= segment_max_x and segment_min_y <= y <= segment_max_y:
                        filtered_points.append(line)

            output_ground = Path(f"{CROPPED_GROUND_PATH}/model{idx}_groundseg{i}.txt")
            with open(output_ground, "w") as out_file:
                out_file.write(header)
                out_file.writelines(filtered_points)

        # model
        for i, boundary in enumerate(boundaries):
            segment_min_x, segment_max_x, segment_min_y, segment_max_y = boundary
            min_z = -1e9  # ground_df['z'].min()
            max_z = 1e9

            OUTPUT_PATH = Path(f"{CROPPED_MODEL_PATH}/model{idx}_segment{i}")
            OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

            crop_command = [
                "colmap",
                "model_cropper",
                "--input_path",
                f"{INPUT_MODEL_PATH}",
                "--output_path",
                f"{OUTPUT_PATH}",
                "--boundary",
                f"{segment_min_x},{segment_min_y},{min_z},{segment_max_x},{segment_max_y},{max_z}",
            ]
            call(crop_command)

            # Convert model to txt format
            TXT_PATH = Path(f"{CROPPED_MODEL_PATH}/model{idx}_segment{i}_txt")
            TXT_PATH.mkdir(parents=True, exist_ok=True)
            cam_pose_file = f"{TXT_PATH}/extract_cam_m{idx}_seg{i}.txt"
            colmap_to_txt(OUTPUT_PATH, TXT_PATH, cam_pose_file)
