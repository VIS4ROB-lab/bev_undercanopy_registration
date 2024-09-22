import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from os import listdir
import shutil
import csv
import cv2
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def get_data_images(model_path):
    """
    This function takes the model path and reads out the quaternion (QW, QX, QY, QZ) and a translation vector (TX, TY, TZ) from the images.txt file

    Parameters:
    -----------
    model_path: Path to model. Model Folder should include: images.txt

    Returns:
    --------
    [QW, QX, QY, QZ, TX, TY, TZ, name]: type List

    With:
    quaternions QW, QX, QY, QZ type: list
    translation vectors TX,TY,TZ type: list
    name: name of image
    """

    images_df = pd.read_csv(model_path / "images.txt", sep=" ", comment="#", header=None, usecols=range(10))
    images_df.columns = ["IMAGE_ID", "QW", "QX", "QY", "QZ", "TX", "TY", "TZ", "CAMERA_ID", "NAME"]

    images_df = images_df.drop(images_df.index[1::2])

    QW = images_df["QW"].tolist()
    QX = images_df["QX"].tolist()
    QY = images_df["QY"].tolist()
    QZ = images_df["QZ"].tolist()

    TX = images_df["TX"].tolist()
    TY = images_df["TY"].tolist()
    TZ = images_df["TZ"].tolist()

    name = images_df["NAME"].tolist()

    return [QW, QX, QY, QZ, TX, TY, TZ, name]


def camera_centers_and_vectors(QW, QX, QY, QZ, TX, TY, TZ):
    """
    The coordinates of the projection/camera center are given by -R^t * T
    R^t is the inverse/transpose of the 3x3 rotation matrix composed from the quaternion
    T is the translation vector
    The local camera coordinate system of an image is defined in a way that the X axis points to the right,
    the Y axis to the bottom, and the Z axis to the front as seen from the image.


    Parameters:
    -----------
    quaternion (QW, QX, QY, QZ), each type lists or as float
    translation vector (TX, TY, TZ), each type lists or as float

    Returns:
    --------
    [centers_x, centers_y, centers_z, x_vec, y_vec, y_vec]: list of all camera centers and list of x and y vector of the camera.
    """
    if type(QW) == list:

        # Camera center
        centers_x = []
        centers_y = []
        centers_z = []

        # Camera orientation
        x_vec_cam = []
        y_vec_cam = []
        z_vec_cam = []

        for idx, _ in enumerate(QW):

            # Translation vector
            t_vec = np.array([TX[idx], TY[idx], TZ[idx]])

            # Quaternion
            quat = np.array([QW[idx], QX[idx], QY[idx], QZ[idx]])
            quat = quat / np.linalg.norm(quat)

            qw = quat[0]
            qx = quat[1]
            qy = quat[2]
            qz = quat[3]

            # Rotation Matrix
            rot_mat = np.zeros((3, 3))

            # Calculate rotation matrix
            rot_mat[0, 0] = 1 - 2 * qy * qy - 2 * qz * qz
            rot_mat[0, 1] = 2 * qx * qy - 2 * qz * qw
            rot_mat[0, 2] = 2 * qx * qz + 2 * qy * qw
            rot_mat[1, 0] = 2 * qx * qy + 2 * qz * qw
            rot_mat[1, 1] = 1 - 2 * qx * qx - 2 * qz * qz
            rot_mat[1, 2] = 2 * qy * qz - 2 * qx * qw
            rot_mat[2, 0] = 2 * qx * qz - 2 * qy * qw
            rot_mat[2, 1] = 2 * qy * qz + 2 * qx * qw
            rot_mat[2, 2] = 1 - 2 * qx * qx - 2 * qy * qy

            R_t = np.transpose(rot_mat)

            camera = -np.dot(R_t, t_vec)

            centers_x.append(camera[0])
            centers_y.append(camera[1])
            centers_z.append(camera[2])

            x_vec = R_t[:, 0]
            y_vec = R_t[:, 1]
            z_vec = R_t[:, 2]

            x_vec_cam.append(x_vec)
            y_vec_cam.append(y_vec)
            z_vec_cam.append(z_vec)

        return [centers_x, centers_y, centers_z, x_vec_cam, y_vec_cam, z_vec_cam]

    t_vec = np.array([TX, TY, TZ])

    quat = np.array([QW, QX, QY, QZ])
    quat = quat / np.linalg.norm(quat)

    qw = quat[0]
    qx = quat[1]
    qy = quat[2]
    qz = quat[3]

    rot_mat = np.zeros((3, 3))

    rot_mat[0, 0] = 1 - 2 * qy * qy - 2 * qz * qz
    rot_mat[0, 1] = 2 * qx * qy - 2 * qz * qw
    rot_mat[0, 2] = 2 * qx * qz + 2 * qy * qw
    rot_mat[1, 0] = 2 * qx * qy + 2 * qz * qw
    rot_mat[1, 1] = 1 - 2 * qx * qx - 2 * qz * qz
    rot_mat[1, 2] = 2 * qy * qz - 2 * qx * qw
    rot_mat[2, 0] = 2 * qx * qz - 2 * qy * qw
    rot_mat[2, 1] = 2 * qy * qz + 2 * qx * qw
    rot_mat[2, 2] = 1 - 2 * qx * qx - 2 * qy * qy

    R_t = np.transpose(rot_mat)

    camera = -np.dot(R_t, t_vec)

    x_vec = R_t[:, 0]
    y_vec = R_t[:, 1]
    z_vec = R_t[:, 2]

    return [camera[0], camera[1], camera[2], x_vec, y_vec, z_vec]


def correct_pitch_and_roll(x_vec_cam_list, y_vec_cam_list, z_vec_cam_list, name_recon_list, df_batches):
    """
    Adapt x,y,z orientation vector by the camera pitch and roll

    Parameters:
    -----------
    x_vec_cam_list, y_vec_cam_list, z_vec_cam_list: list of x,y and z vectors of camera orientations
    name_recon_list: list of image names extracted from images.txt
    df_batches: dataframe of image data stored in Images_LV95_batches.csv

    Returns:
    --------
    x_vec_new, y_vec_new, z_vec_new: lists of adapted x,y,z vectors
    """

    names_csv_list = df_batches["name"].tolist()
    gimball_pitch = df_batches["gimball_pitch"].tolist()
    gimball_roll = df_batches["gimball_roll"].tolist()

    x_vec_new = []
    y_vec_new = []
    z_vec_new = []

    for idx, tar_name in enumerate(name_recon_list):
        x_vec = x_vec_cam_list[idx]
        y_vec = y_vec_cam_list[idx]
        z_vec = z_vec_cam_list[idx]

        # Find position of corresponding name_csv_list
        index_df = names_csv_list.index(tar_name)

        # Get corresponding pitch and roll angles
        gim_pitch = float(gimball_pitch[index_df])
        gim_roll = float(gimball_roll[index_df])

        pitch = -(gim_pitch)
        roll = -(gim_roll)

        # Rotation matrices
        R_t = np.array([[x_vec[0], y_vec[0], z_vec[0]], [x_vec[1], y_vec[1], z_vec[1]], [x_vec[2], y_vec[2], z_vec[2]]])
        C_pitch = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(pitch), -np.sin(pitch)], [0.0, np.sin(pitch), np.cos(pitch)]])
        C_roll = np.array([[np.cos(roll), -np.sin(roll), 0.0], [np.sin(roll), np.cos(roll), 0.0], [0.0, 0.0, 1.0]])

        # Correct for pitch and roll of image
        rot_1 = R_t @ C_pitch
        rot_2 = rot_1 @ C_roll

        x_vec_new.append(rot_2[:, 0])
        y_vec_new.append(rot_2[:, 1])
        z_vec_new.append(rot_2[:, 2])

    return x_vec_new, y_vec_new, z_vec_new


def pixel_to_world_ortho(pixel, width, height, res_x, res_y):
    """
    This Function was created by: Junyuan Cui
    """
    if isinstance(pixel, list):
        return [[(i[0] - res_x / 2) * width / res_x, (res_y - i[1]) * height / res_y] for i in pixel]
    return [(pixel[0] - res_x / 2) * width / res_x, (res_y - pixel[1]) * height / res_y]


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # -                           ARGPARSE                             -
    # ------------------------------------------------------------------

    # Argparse: Create parser
    parser = argparse.ArgumentParser()
    # Argparse: Add arguments
    parser.add_argument(dest="project_path", help="Path to the project folder including Models/Model_i folders")
    parser.add_argument(
        dest="bird_eye_view",
        nargs="+",
        help="Path to folder that contains bird eye view (BEV) predictions. If multiple provide them in the order of the models they belong to (Path/to/Model_0 Path/to/Model_2 ...)",
    )
    parser.add_argument(
        "-cluster_rad",
        "--cluster_rad",
        default="2.0",
        help="Max radius at which trees should be considered as the same tree. Default: 2.0",
    )
    # Argparse: Parse arguments
    args = parser.parse_args()

    IMAGES_LV95_BATCHES = f"{args.project_path}/Images_LV95_batches.csv"
    PROJECT_PATH = Path(args.project_path)

    # ------------------------------------------------------------------
    # -                  Calculate Tree Positions                     -
    # ------------------------------------------------------------------

    # Set cluster-radius
    cluster_radius = float(args.cluster_rad)

    # Check if multiple bird eye view (BEV) image folders exist
    if len(args.bird_eye_view) == 1:
        multiple_BEV_folders = False
        BIRD_EYE_PATH = Path(args.bird_eye_view[0])

        bir_im_names = listdir(BIRD_EYE_PATH)
        bir_im_names_clean = []
        for image_str in bir_im_names:
            str_split = image_str.split(".")
            im_clean = str_split[0]
            bir_im_names_clean.append(im_clean)
    else:
        multiple_BEV_folders = True

    # Read number of batches/models
    df_batches = pd.read_csv(IMAGES_LV95_BATCHES)
    max_batch = df_batches["batch"].max()
    nr_batches = int(max_batch) + 1

    # Loop through models/batches
    for idx in range(nr_batches):
        print(f"Calculating tree positions model_{idx}...")

        if multiple_BEV_folders:
            BIRD_EYE_PATH = Path(args.bird_eye_view[idx])

            bir_im_names = listdir(BIRD_EYE_PATH)
            bir_im_names_clean = []
            for image_str in bir_im_names:
                str_split = image_str.split(".")
                im_clean = str_split[0]
                bir_im_names_clean.append(im_clean)

        # Create folder for tree positions
        TREE_POS_LV95 = Path(f"{PROJECT_PATH}/Models/Model_{idx}/tree_positions_shifted_LV95")  # gt BEV
        # TREE_POS_LV95 = Path(f'{PROJECT_PATH}/Models/Model_{idx}/tree_positions_shifted_LV95_original')  # predicted BEV
        if TREE_POS_LV95.exists():
            shutil.rmtree(TREE_POS_LV95)
        TREE_POS_LV95.mkdir(parents=True, exist_ok=True)

        # Create tree position file
        tree_positions_file = f"{TREE_POS_LV95}/candidate_tree_positions.txt"
        with open(tree_positions_file, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["x", "y", "z"])

        # Get image data: camera centers and camera orientation vectors
        PATH_MODEL_LV95_TXT = Path(f"{PROJECT_PATH}/Models/Model_{idx}/shifted_LV95_model_txt")
        qw, qx, qy, qz, tx, ty, tz, name = get_data_images(PATH_MODEL_LV95_TXT)
        cam_x, cam_y, cam_z, x_vec_cam_list, y_vec_cam_list, z_vec_cam_list = camera_centers_and_vectors(
            qw, qx, qy, qz, tx, ty, tz
        )
        x_vec_list, y_vec_list, z_vec_list = correct_pitch_and_roll(
            x_vec_cam_list, y_vec_cam_list, z_vec_cam_list, name, df_batches
        )  # correct x,y,z direction of camera image with pitch and roll of drone gimball

        x_points_tree = []
        y_points_tree = []
        z_points_tree = []

        # Go through images and check if bird-eye-view exists
        for n, model_im_names in enumerate(name):
            # print(f'name: {model_im_names}')
            split_model_im_name = model_im_names.split(".")
            bird_im_name = split_model_im_name[0]

            # If bird eye-view exists find tree positions and map them to 3D
            if bird_im_name in bir_im_names_clean:
                # Get image path in bird eye view path
                ind_image = bir_im_names_clean.index(bird_im_name)
                image = bir_im_names[ind_image]
                bird_im_path = f"{BIRD_EYE_PATH}/{image}"

                # Find tree blobs in bird eye view
                # Code from: Junyuan Cui START:
                # -------------------------------------------
                img = cv2.imread(f"{bird_im_path}")
                img = cv2.resize(img, dsize=(480, 480))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.bitwise_not(gray)
                params = cv2.SimpleBlobDetector_Params()
                params.filterByInertia = True
                params.minInertiaRatio = 0.01
                ver = (cv2.__version__).split(".")
                if int(ver[0]) < 3:
                    detector = cv2.SimpleBlobDetector(params)
                else:
                    detector = cv2.SimpleBlobDetector_create(params)
                keypoints = detector.detect(gray)

                # # Show found keypoints
                # img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                # cv2.imshow('Image with Keypoints', img_with_keypoints)
                # cv2.waitKey(0)

                points = []
                for k in keypoints:
                    if k.size > 55:
                        points.append(k.pt)
                centers = pixel_to_world_ortho(points, width=30.0, height=30.0, res_x=480, res_y=480)
                # -------------------------------------------
                # Code from: Junyuan Cui END

                # plt.figure(figsize=(12, 8))
                # Now that we have each center in 2D coordinates of image --> map to 3D coordinate system
                image_3D_points = []
                for _, blob_center in enumerate(centers):

                    cam_center = np.array([cam_x[n], cam_y[n], cam_z[n]])

                    x_vec = np.array(x_vec_list[n])
                    z_vec = np.array(z_vec_list[n])

                    x_shift = blob_center[0] * x_vec
                    z_shift = blob_center[1] * z_vec

                    vec_1 = np.add(cam_center, x_shift)
                    point_3D = np.add(vec_1, z_shift)

                    x_points_tree.append(point_3D[0])
                    y_points_tree.append(point_3D[1])
                    z_points_tree.append(point_3D[2])

                    plt.scatter(point_3D[0], point_3D[1], s=800)

                    # Save candidate tree positions to txt file
                    with open(tree_positions_file, "a", newline="") as file:
                        writer = csv.writer(file)
                        writer.writerow([point_3D[0], point_3D[1], point_3D[2]])
                # plt.xlabel('X-axis [m]', fontsize= 32)
                # plt.ylabel('Y-axis [m]', fontsize= 30)
                # plt.xticks(fontsize= 22)
                # plt.yticks(fontsize=20)
                # plt.title('Top View Tree Positions', fontsize = 40)
                # plt.grid(True)
                # plt.show()

        if len(x_points_tree) > 0:  # Check if any trees where found

            # Get one point per tree from candidate tree position points:

            # Group points with DBSCAN
            data_trees = np.array([x_points_tree, y_points_tree]).T
            epsilon = cluster_radius
            clustering = DBSCAN(eps=epsilon, min_samples=3)  # 4 #3
            labels = clustering.fit_predict(data_trees)

            # Save clustered tree positions to txt file
            tree_clusters_file = f"{TREE_POS_LV95}/clustered_tree_positions.txt"
            clustered_data = zip(x_points_tree, y_points_tree, z_points_tree, labels)
            with open(tree_clusters_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["x", "y", "z", "label"])
                writer.writerows(clustered_data)

            # Get average tree positions per tree (cluster)
            x_array = np.array(x_points_tree)
            y_array = np.array(y_points_tree)
            z_array = np.array(z_points_tree)

            max_label = max(labels)
            mean_tree_x = []
            mean_tree_y = []
            mean_tree_z = []

            for lab in range(max_label + 1):
                indexes = np.where(np.array(labels) == lab)[0]

                x_mean = np.mean(np.array(x_array)[indexes])
                y_mean = np.mean(np.array(y_array)[indexes])
                z_mean = np.mean(np.array(z_array)[indexes])

                mean_tree_x.append(x_mean)
                mean_tree_y.append(y_mean)
                mean_tree_z.append(z_mean)

            # Save found tree position to txt file
            clustered_data = zip(mean_tree_x, mean_tree_y, mean_tree_z)
            mean_trees_file = f"{TREE_POS_LV95}/tree_positions.txt"
            with open(mean_trees_file, "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["x", "y", "z"])
                writer.writerows(clustered_data)

            print(f"Saved tree positions at: {mean_trees_file}")

        else:

            print(f"No trees found for model_{idx}")

    print("Done!")
