import argparse
from pathlib import Path
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.spatial.transform import Rotation as R
import pandas as pd


def transform_pose(qw, qx, qy, qz, x, y, z, transformation_matrix):
    # ! camera pose from colmap(camera-to-world), inverse first
    # Convert quaternion to rotation matrix
    rotation = R.from_quat([qx, qy, qz, qw])
    rotation_matrix = rotation.inv().as_matrix()

    transformed_rotation_matrix = np.dot(rotation_matrix, transformation_matrix[:3, :3])
    # Convert the rotation matrix back to quaternion
    transformed_rotation = R.from_matrix(transformed_rotation_matrix)
    transformed_quaternion = transformed_rotation.inv().as_quat()

    # Apply transformation to position
    position = np.array([x, y, z, 1])
    transformed_position = np.dot(position, transformation_matrix)[:3]

    return (*transformed_quaternion[[3, 0, 1, 2]], *transformed_position)  # *unpack


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # -                           ARGPARSE                             -
    # ------------------------------------------------------------------

    # Argparse: Create parser
    parser = argparse.ArgumentParser()
    # Argparse: Add arguments
    parser.add_argument(dest="project_path", help="Path to the project folder, containing Models/Model_i folders")

    # Argparse: Parse arguments
    args = parser.parse_args()

    PROJECT_PATH = f"{args.project_path}"
    BATCH_ALIGN_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/3_batch_align")
    BATCH_INIT_PATH = Path(f"{BATCH_ALIGN_PATH}/Init_files")
    BATCH_PATH = Path(f"{BATCH_ALIGN_PATH}/Batches")  # /Cropped_model1
    # ALLSEG_MODELS_PATH = Path(f'{BATCH_ALIGN_PATH}/Horizontal_correct/Model1') # output batches

    # OUTPUT_PATH = Path(f'{BATCH_ALIGN_PATH}/Output/Cam_poses_m1')
    # OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    CAMERA_COMPARE_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/Camera_poses_compare")

    IMAGES_LV95_BATCHES = f"{args.project_path}/Images_LV95_batches.csv"
    df_batches = pd.read_csv(IMAGES_LV95_BATCHES)
    max_batch = df_batches["batch"].max()
    nr_models = int(max_batch) + 1

    # -------------------- Read boundaries from file ----------------------------
    for idx in range(1, nr_models):
        ALLSEG_MODELS_PATH = Path(f"{BATCH_ALIGN_PATH}/Horizontal_correct/Model{idx}")  # output batches
        OUTPUT_PATH = Path(f"{BATCH_ALIGN_PATH}/Output/Cam_poses_m{idx}")
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        boundaries = []
        boundaries_original = []
        boundary_file_path = f"{BATCH_PATH}/bounding_boxes_m{idx}.txt"
        with open(boundary_file_path, "r") as file:
            next(file)
            nr_segs = 0
            for line in file:
                segments = line.strip().split()
                boundary = [float(segment) for segment in segments]
                boundaries.append(boundary)
                nr_segs = nr_segs + 1

        boundaries_original = copy.deepcopy(boundaries)
        # replace seg_min_x == min all_segs_xmin to -inf
        all_xmin, all_xmax, all_ymin, all_ymax = zip(
            *[(boundary[0], boundary[1], boundary[2], boundary[3]) for boundary in boundaries]
        )
        outer_min_x = min(all_xmin)
        outer_max_x = max(all_xmax)
        outer_min_y = min(all_ymin)
        outer_max_y = max(all_ymax)

        for i, boundary in enumerate(boundaries):
            # seg_min_x, seg_max_x, seg_min_y, seg_max_y = boundary
            if boundary[0] <= outer_min_x + 3:
                boundary[0] = float("-inf")
            if boundary[1] >= outer_max_x - 3:
                boundary[1] = float("inf")
            if boundary[2] <= outer_min_y + 3:
                boundary[2] = float("-inf")
            if boundary[3] >= outer_max_y - 3:
                boundary[3] = float("inf")

        # --------------------- read camera poses data ---------------------------

        camera_poses = {}
        for i in range(nr_segs):
            SEG_MODEL_FILE = Path(f"{ALLSEG_MODELS_PATH}/Seg{i}/Seg{i}_Model_txt/extracted_cam{i}.txt")
            with open(SEG_MODEL_FILE, "r") as file:
                next(file)
                for line in file:
                    parts = line.strip().split()
                    filename = parts[0]
                    pose = [float(x) for x in parts[1:]]
                    if filename not in camera_poses:
                        camera_poses[filename] = {"pose": pose, "batches": [i]}
                    else:
                        camera_poses[filename]["batches"].append(i)

        # -------------------- cameras assignment ---------------------------------
        # # v-
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')

        for camera, data in camera_poses.items():
            # camera coordinates
            qw, qx, qy, qz, x, y, z = data["pose"]
            data["coordinates"] = (x, y, z)
            # forward_direction = np.array([2*qx*qz + 2*qy*qw, 2*qy*qz - 2*qx*qw, 1 - 2*qx**2 - 2*qy**2])
            qw_inv, qx_inv, qy_inv, qz_inv = qw, -qx, -qy, -qz
            forward_direction = np.array(
                [
                    2 * qx_inv * qz_inv + 2 * qy_inv * qw_inv,
                    2 * qy_inv * qz_inv - 2 * qx_inv * qw_inv,
                    1 - 2 * qx_inv**2 - 2 * qy_inv**2,
                ]
            )

            # # v-
            forward_direction /= np.linalg.norm(forward_direction)
            data["forward_direction"] = forward_direction
            # ax.scatter(x, y, z, color='blue', s=50)
            # ax.quiver(x, y, z, forward_direction[0], forward_direction[1], forward_direction[2], length=4, color='red')

            in_boundaries = []
            # assignment logic
            if len(data["batches"]) == 1:
                data["assigned_batch"] = data["batches"][0]
            else:
                for i, boundary in enumerate(boundaries):
                    min_x, max_x, min_y, max_y = boundary
                    # check bounding boxes the cam falls in
                    if min_x <= x <= max_x and min_y <= y <= max_y:
                        in_boundaries.append(i)

                if len(in_boundaries) == 1:
                    data["assigned_batch"] = in_boundaries[0]
                elif len(in_boundaries) > 1:
                    # data['assigned_batch'] = random.choice(in_boundaries)
                    # based on its forward-facing direction
                    boundary_centers = [
                        (0.5 * (min_x + max_x), 0.5 * (min_y + max_y))
                        for min_x, max_x, min_y, max_y in [boundaries_original[i] for i in in_boundaries]
                    ]
                    camera_position = np.array([x, y])

                    angles = [
                        np.arccos(
                            np.clip(
                                np.dot(forward_direction[:2], np.array(center) - camera_position)
                                / (
                                    np.linalg.norm(forward_direction[:2])
                                    * np.linalg.norm(np.array(center) - camera_position)
                                ),
                                -1.0,
                                1.0,
                            )
                        )
                        for center in boundary_centers
                    ]
                    closest_boundary_index = in_boundaries[np.argmin(angles)]
                    data["assigned_batch"] = closest_boundary_index
                else:  # find nearest boundary
                    all_boundary_centers = [
                        (0.5 * (min_x + max_x), 0.5 * (min_y + max_y))
                        for min_x, max_x, min_y, max_y in boundaries_original
                    ]
                    camera_position_2d = np.array([x, y])
                    acute_boundaries = []
                    center_distance = []

                    for i, center in enumerate(all_boundary_centers):
                        center_vector = np.array(center) - camera_position_2d
                        angle = np.arccos(
                            np.clip(
                                np.dot(forward_direction[:2], center_vector)
                                / (np.linalg.norm(forward_direction[:2]) * np.linalg.norm(center_vector)),
                                -1.0,
                                1.0,
                            )
                        )
                        center_distance.append((i, np.linalg.norm(center_vector)))
                        if angle < np.pi / 2:  # Acute angle
                            acute_boundaries.append((i, np.linalg.norm(center_vector)))  # Store index and distance

                    if acute_boundaries:
                        # Choose the nearest boundary among those with acute angles
                        closest_boundary_index, _ = min(acute_boundaries, key=lambda x: x[1])
                        data["assigned_batch"] = closest_boundary_index
                    else:
                        closest_boundary_index, _ = min(center_distance, key=lambda x: x[1])
                        data["assigned_batch"] = closest_boundary_index

        # # v-
        # ax.set_xlabel('X Axis')
        # ax.set_ylabel('Y Axis')
        # ax.set_zlabel('Z Axis')
        # ax.set_title('Camera Forward-Facing Vectors')
        # plt.show()

        # -------------------- vidual --------
        colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black", "orange"]
        # Ensure you have enough colors for the number of clusters
        fig, ax = plt.subplots()

        for i, boundary in enumerate(boundaries_original):
            min_x, max_x, min_y, max_y = boundary
            rect = plt.Rectangle(
                (min_x, min_y), max_x - min_x, max_y - min_y, linewidth=2, edgecolor=colors[i], facecolor="none"
            )
            ax.add_patch(rect)

        # Plot cameras and their forward-facing vectors
        for camera, data in camera_poses.items():
            x, y, z = data["coordinates"]
            forward_direction = data["forward_direction"][:2]  # Considering only x and y for 2D
            assigned_batch = data["assigned_batch"]
            color = colors[
                assigned_batch % len(colors)
            ]  # Use modulo to cycle through colors if there are more batches than colors

            ax.scatter(x, y, color=color, s=25)
            ax.quiver(x, y, forward_direction[0], forward_direction[1], color=color, scale=15)

        ax.set_xlabel("X Axis", fontsize=16)
        ax.set_ylabel("Y Axis", fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.set_title("Camera Forward-Facing Vectors and Bounding Boxes", fontsize=16)
        plt.show()

        # ---------------------------- save ---------------------------------------
        header = "filename Assigned_Batch qw qx qy qz x y z\n"
        batch_files = {i: open(f"{OUTPUT_PATH}/batch{i}_cameras.txt", "w") for i in range(nr_segs)}
        for file in batch_files.values():
            file.write(header)

        all_cameras_file = open(f"{OUTPUT_PATH}/batch_align_cameras_m1.txt", "w")
        all_cameras_file.write(header)

        cam_compare_file = open(f"{CAMERA_COMPARE_PATH}/eval_m{idx}_batch_aligned.txt", "w")
        cam_compare_file.write("filename qw qx qy qz x y z\n")

        for camera, data in camera_poses.items():
            batch_file = batch_files[data["assigned_batch"]]
            batch_file.write(f'{camera} {data["assigned_batch"]} {" ".join(map(str, data["pose"]))}\n')
            all_cameras_file.write(
                f'{camera} {data["assigned_batch"]} {" ".join(map(str, data["pose"]))}\n'
            )  # {" ".join(f"{v:.6f}" for v in data["pose"])}\n')
            cam_compare_file.write(
                f'{camera} {" ".join(map(str, data["pose"]))}\n'
            )  # {" ".join(f"{v:.6f}" for v in data["pose"])}\n')

        for file in batch_files.values():
            file.close()
        all_cameras_file.close()

        # --------------------------------------------------------------------------------
        # ------                        extract cam compare                         ------
        # --------------------------------------------------------------------------------
        # for idx in range(1,nr_models):
        TRANS_FILE = f"{BATCH_INIT_PATH}/trans_m1_xy.txt"
        transformation_matrix = np.loadtxt(TRANS_FILE)
        inverse_transformation_matrix = np.linalg.inv(transformation_matrix)

        # -- input batch
        # INPUT_CAM_FILE = f'{BATCH_INIT_PATH}/Models/model{idx}_txt/batch_init_cam_m{idx}.txt'
        INPUT_CAM_FILE = f"{CAMERA_COMPARE_PATH}/cam_eval_m{idx}_loop.txt"
        with open(INPUT_CAM_FILE, "r") as file:
            init_cam_lines = file.readlines()

        init_cam_data = {}
        for line in init_cam_lines[1:]:  # Skipping header
            parts = line.strip().split()
            filename = parts[0]
            pose = parts[1:]  # Storing as string list
            init_cam_data[filename] = " ".join(pose)

        # -- output batch
        FINAL_BATCH_PATH = Path(f"{BATCH_ALIGN_PATH}/Output/Cam_poses_m{idx}")
        for i in range(nr_segs):
            FINAL_CAM_FILE = f"{FINAL_BATCH_PATH}/batch{i}_cameras.txt"
            with open(FINAL_CAM_FILE, "r") as file:
                lines = file.readlines()

            final_cam_data = []
            filtered_init_cam_data = []
            for line in lines[1:]:
                parts = line.strip().split()
                filename, pose_data = parts[0], " ".join(parts[2:])
                final_cam_data.append(f"{filename} {pose_data}\n")

                # Filter corresponding initial camera pose
                if filename in init_cam_data:
                    filtered_init_cam_data.append(f"{filename} {init_cam_data[filename]}\n")
                # # Transform the corresponding initial camera pose
                # if filename in init_cam_data:
                #     init_qw, init_qx, init_qy, init_qz, init_x, init_y, init_z = init_cam_data[filename]
                #     transformed_init_pose = transform_pose(init_qw, init_qx, init_qy, init_qz, init_x, init_y, init_z, transformation_matrix)
                #     transformed_init_data.append(f"{filename} {' '.join(map(str, transformed_init_pose))}\n")

            # save
            header = "filename qw qx qy qz x y z\n"
            OUTPUT_FINAL_BATCH_CAM_PATH = Path(f"{CAMERA_COMPARE_PATH}/Batch_final/model{idx}")
            OUTPUT_FINAL_BATCH_CAM_PATH.mkdir(parents=True, exist_ok=True)
            trans_back_final_batch_cam_file = f"{OUTPUT_FINAL_BATCH_CAM_PATH}/model{idx}_final_batch{i}.txt"
            with open(trans_back_final_batch_cam_file, "w") as file:
                file.write(header)
                file.writelines(final_cam_data)

            # Save filtered init cam data
            OUTPUT_INIT_BATCH_CAM_PATH = Path(f"{CAMERA_COMPARE_PATH}/Batch_init/model{idx}")
            OUTPUT_INIT_BATCH_CAM_PATH.mkdir(parents=True, exist_ok=True)
            trans_back_init_batch_cam_file = f"{OUTPUT_INIT_BATCH_CAM_PATH}/model{idx}_init_batch{i}.txt"
            with open(trans_back_init_batch_cam_file, "w") as file:
                file.write(header)
                file.writelines(filtered_init_cam_data)

            print(f"final_batches of model{idx}_seg{i} successfully transformed back")
