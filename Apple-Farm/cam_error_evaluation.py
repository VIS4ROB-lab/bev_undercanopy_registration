import argparse
from pathlib import Path
import numpy as np
import math
import quaternion
from typing import Tuple, Dict, List
import json
import pandas as pd

import kapture
from kapture import Trajectories, PoseTransform
from kapture.algo.pose_operations import pose_transform_distance
import matplotlib.pyplot as plt
import os


class CameraPoses(Dict[str, PoseTransform]):
    """
    CameraPoses class storing PoseTransforms indexed by filename.
    """

    def __setitem__(self, key: str, value: PoseTransform):
        if not isinstance(key, str):
            raise TypeError("Key must be a string representing the filename")
        if not isinstance(value, PoseTransform):
            raise TypeError("Value must be an instance of PoseTransform")
        super().__setitem__(key, value)

    def __getitem__(self, key: str) -> PoseTransform:
        if not isinstance(key, str):
            raise TypeError("Key must be a string representing the filename")
        return super().__getitem__(key)


def read_filename_pose(path: Path) -> Tuple[List[str], CameraPoses]:
    keys = []
    poses = CameraPoses()
    with open(path, "r") as fid:
        for line in fid:
            line = line.strip("\n")
            parts = line.split(" ")
            if parts[0] == "filename":
                continue
            filename, qw, qx, qy, qz, tx, ty, tz = parts
            pose = PoseTransform(np.array([qw, qx, qy, qz], dtype=float), np.array([tx, ty, tz], dtype=float))
            poses[filename] = pose
            keys.append((filename))
    return keys, poses


def quaternion_to_rotation_matrix(quaternion):
    qw, qx, qy, qz = quaternion
    R = np.array(
        [
            [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
            [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qx * qw],
            [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx**2 - 2 * qy**2],
        ]
    )
    return R


def calculate_rotation_error(quaternion_camera, quaternion_gt):
    R_camera = quaternion_to_rotation_matrix(quaternion_camera.components)
    R_gt = quaternion_to_rotation_matrix(quaternion_gt.components)

    forward_axis_camera = R_camera[:, 2]
    forward_axis_gt = R_gt[:, 2]

    cos_angle = np.dot(forward_axis_camera, forward_axis_gt) / (
        np.linalg.norm(forward_axis_camera) * np.linalg.norm(forward_axis_gt)
    )
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    angle_degrees = np.degrees(angle)
    return angle_degrees


def pose_transform_distance(
    pose_a: kapture.PoseTransform, pose_b: kapture.PoseTransform, coincident: bool
) -> Tuple[float, float]:
    """
    get translation and rotation distance between two PoseTransform
    :return: (position_distance, rotation_distance in rad), can be nan is case of invalid comparison
    """
    # handle NoneType with try expect blocks
    try:
        translation_distance = np.linalg.norm(pose_a.t - pose_b.t)
    except TypeError:
        translation_distance = math.nan

    try:
        # if coincident == 0:
        #     rotation_distance = quaternion.rotation_intrinsic_distance(pose_a.r.inverse(), pose_b.r) # for colmap & gt
        # if coincident == 1:
        #     rotation_distance = quaternion.rotation_intrinsic_distance(pose_a.r, pose_b.r) # for colmap & colmap
        if coincident == 0:
            rotation_distance = calculate_rotation_error(pose_a.r.inverse(), pose_b.r)  # for colmap & gt
        if coincident == 1:
            rotation_distance = calculate_rotation_error(pose_a.r, pose_b.r)  # for colmap & colmap
    except TypeError:
        rotation_distance = math.nan
    return translation_distance, rotation_distance


def plot_recalls(threshs, recalls, threshs_t, recall_t, threshs_r, recall_r, output_plot_path):
    plt.figure(figsize=(12, 4))
    # combined recall
    plt.subplot(1, 3, 1)
    plt.bar(range(len(recalls)), recalls, tick_label=[f"{th[0]}, {th[1]}" for th in threshs])
    plt.ylim(0, 0.7)
    plt.xlabel("Thresholds (R, T)")
    plt.ylabel("Recall")
    plt.title("Recall for Different Thresholds")

    # Plot for recall_t
    plt.subplot(1, 3, 2)
    plt.bar(range(len(recall_t)), recall_t, tick_label=[str(th) for th in threshs_t])
    plt.ylim(0, 0.9)
    plt.xlabel("T Thresholds")
    plt.ylabel("Recall")
    plt.title("Recall T Above Thresholds")

    # Plot for recall_r
    plt.subplot(1, 3, 3)
    plt.bar(range(len(recall_r)), recall_r, tick_label=[str(th) for th in threshs_r])
    plt.ylim(0, 1)
    plt.xlabel("R Thresholds")
    plt.ylabel("Recall")
    plt.title("Recall R Above Thresholds")

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.savefig(f"{output_plot_path}")
    plt.show()


def plot_boxplot(error, output_boxplot_file, y_lim=None, whiskers=[25, 75]):
    plt.boxplot(error, showfliers=True, whis=whiskers)
    if y_lim:
        if y_lim == True:
            Q1 = np.percentile(error, 25)
            Q3 = np.percentile(error, 75)
            IQR = Q3 - Q1
            plt.ylim(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
        else:
            plt.ylim(y_lim)

    # Get the statistics
    Q1 = np.percentile(error, 25)
    Q3 = np.percentile(error, 75)
    median = np.median(error)
    # Annotate the median value on the plot
    plt.text(1, median, f"Median: {median:.2f}", verticalalignment="center", horizontalalignment="right")
    # plt.text(1, Q1, f'25th percentile: {Q1:.2f}', verticalalignment='center', horizontalalignment='right')
    # plt.text(1, Q3, f'75th percentile: {Q3:.2f}', verticalalignment='center', horizontalalignment='right')

    plt.title(f"Box Plot_{os.path.basename(output_boxplot_file)}")
    plt.ylabel("Values")
    plt.savefig(f"{output_boxplot_file}")
    plt.show()


def evaluate(
    cam_pose_gt: Path,
    cam_pose_to_eval: Path,
    output_path: Path,
    coincident: bool,
    output_plot_recall=None,
    output_boxplot_name=None,
):  # gt, to eval,

    keys, poses_eval = read_filename_pose(cam_pose_to_eval)
    keys_gt, poses_gt = read_filename_pose(cam_pose_gt)

    err_r, err_t = [], []
    err_t_save = []
    outlier_eval_save, outlier_gt_save = [], []
    for key in keys:
        if key in keys_gt:
            T_w2c_gt = poses_gt[key]

            dt, dr = pose_transform_distance(poses_eval[key], T_w2c_gt, coincident)
            # dr = np.rad2deg(dr)

            err_r.append(dr)
            err_t.append(dt)

            # analysis & find outliers
            err_t_save.append([key, dt, dt > 1])
            if dt > 1:
                qw, qx, qy, qz = poses_eval[key].r.components
                tx, ty, tz = poses_eval[key].t
                tx_str = str(tx[0]) if tx else "nan"
                ty_str = str(ty[0]) if ty else "nan"
                tz_str = str(tz[0]) if tz else "nan"
                outlier_eval_save.append(f"{key} {qw} {qx} {qy} {qz} {tx_str} {ty_str} {tz_str}")

                qw, qx, qy, qz = poses_gt[key].r.components
                tx, ty, tz = poses_gt[key].t
                tx_str = str(tx[0]) if tx else "nan"
                ty_str = str(ty[0]) if ty else "nan"
                tz_str = str(tz[0]) if tz else "nan"
                outlier_gt_save.append(f"{key} {qw} {qx} {qy} {qz} {tx_str} {ty_str} {tz_str}")
        else:
            dt = np.nan
            err_t_save.append([key, dt, ""])

    err_r = np.stack(err_r)
    err_t = np.stack(err_t)
    np.savetxt(f"{output_boxplot_name}_t.txt", err_t, fmt="%s", delimiter=" ")
    np.savetxt(f"{output_boxplot_name}_r.txt", err_r, fmt="%s", delimiter=" ")

    # # analysis
    # err_t_save = np.stack(err_t_save)
    # err_t_save_array = np.array(err_t_save, dtype=object)
    # np.savetxt('2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/analyse.txt', err_t_save_array, fmt='%s', delimiter=' ')
    # pd.DataFrame(outlier_eval_save).to_csv('2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/analyse_eval.txt', index=False)
    # pd.DataFrame(outlier_gt_save).to_csv('2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/analyse_gt.txt', index=False)
    # # print(err_t)
    # # print(err_r)

    # Calculate RMSE(Root Mean Square Error) for translation
    rmse_t = np.sqrt(np.mean(np.square(err_t)))
    # Calculate MRE(Mean Rotation Error) for rotation
    mre_r = np.mean(err_r)
    # Calculate recall
    threshs = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1.0, 1.0), (1.25, 1.25)]
    recalls = [np.mean((err_r > th_r) & (err_t > th_t)) for th_r, th_t in threshs]
    threshs_t = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    recall_t = [np.mean((err_t > th_t)) for th_t in threshs_t]
    threshs_r = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    recall_r = [np.mean((err_r > th_r)) for th_r in threshs_r]

    results = {
        "RMSE_t": rmse_t,
        "MRE_r": mre_r,
        "recall": recalls,
        "Rt_thresholds": threshs,
        "recall_t_above": recall_t,
        "T_thresholds": threshs_t,
        "recall_r_above": recall_r,
        "R_thresholds": threshs_r,
    }
    print("Results:", results)

    with open(output_path, "w") as file:
        json.dump(results, file, indent=2)

    # plot
    if output_plot_recall:
        plot_recalls(threshs, recalls, threshs_t, recall_t, threshs_r, recall_r, output_plot_recall)
    if output_boxplot_name:
        output_boxplot_T = f"{output_boxplot_name}_t.png"
        output_boxplot_R = f"{output_boxplot_name}_r.png"
        y_lim_t = True  # [0,1.75] #True #[0,5] #None
        y_lim_r = True  # [0,1.75] #True #[0,5] #None
        whiskers_t = [10, 90]
        whiskers_r = [10, 90]
        plot_boxplot(err_t, output_boxplot_T, y_lim_t, whiskers_t)  # (err_t, output_boxplot_T, y_lim_t, whiskers_t)
        plot_boxplot(err_r, output_boxplot_R, y_lim_r, whiskers_r)  # (err_r, output_boxplot_R, y_lim_r, whiskers_r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=Path, default=Path("data/"))
    parser.add_argument("--output_path", type=Path, default=Path("./outputs/"))
    args = parser.parse_args()
    DATA_PATH = f"{args.data_path}"
    OUTPUT_PLOT = Path(f"{args.output_path}/Figures")
    OUTPUT_PLOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_MIDDLE = Path(f"{args.output_path}/middle_errors")
    OUTPUT_MIDDLE.mkdir(parents=True, exist_ok=True)
    OUTPUT_MIDDLE_PLOT = Path(f"{args.output_path}/Figures/middle_plot")
    OUTPUT_MIDDLE_PLOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_BATCH = Path(f"{args.output_path}/batches_errors")
    OUTPUT_BATCH.mkdir(parents=True, exist_ok=True)
    OUTPUT_BATCH_PLOT = Path(f"{args.output_path}/Figures/batches_plot")
    OUTPUT_BATCH_PLOT.mkdir(parents=True, exist_ok=True)

    nr_models = 2 #set the number of models

    # reconstruction(lower bound)
    for idx in range(1, nr_models):
        evaluate(
            f'{DATA_PATH}/cam_gt_m{idx}.txt',
            f'{DATA_PATH}/cam_gt_extract_m{idx}.txt',
            f'{args.output_path}/gt_errors{idx}.txt', 0,
            f'{OUTPUT_PLOT}/reconstruction_recalls_m{idx}.png',
            f'{OUTPUT_PLOT}/reconstruction_boxplot_m{idx}')

    # Colmap Init
    for idx in range(1, nr_models):
        evaluate(
            f'{DATA_PATH}/cam_gt_m{idx}.txt',
            f'{DATA_PATH}/cam_colmap_init_m{idx}.txt',
            f'{args.output_path}/colmapinit_cam_errors_m{idx}_gt.txt', 0,
            f'{OUTPUT_PLOT}/co_init_recalls_m{idx}.png',
            f'{OUTPUT_PLOT}/co_init_boxplot_m{idx}')

    # Loop Init (after initial ICP_swisstopo)
    for idx in range(1, nr_models):
        evaluate(
            f'{DATA_PATH}/cam_gt_m{idx}.txt',
            f'{DATA_PATH}/cam_eval_m{idx}_initial_aligned.txt',
            f'{args.output_path}/initial_aligned_cam_errors_m{idx}_gt.txt', 0,
            f'{OUTPUT_PLOT}/initial_aligned_recalls_m{idx}.png',
            f'{OUTPUT_PLOT}/initial_aligned_boxplot_m{idx}')

    # after_loop
    for idx in range(1, nr_models):
        evaluate(
            f'{DATA_PATH}/cam_gt_m{idx}.txt',
            f'{DATA_PATH}/cam_eval_m{idx}_loop.txt',
            f'{args.output_path}/loop_cam_errors_m{idx}_gt.txt', 0,
            f'{OUTPUT_PLOT}/loop_recalls_m{idx}.png',
            f'{OUTPUT_PLOT}/loop_boxplot_m{idx}')

    # loop process
    for i in [1,2,5]: #1,2,5,10
        evaluate(
            f'{DATA_PATH}/cam_gt_m1.txt',
            f'{DATA_PATH}/middles/cam_eval_m1_middle{i}_V.txt',
            f'{OUTPUT_MIDDLE}/cam_errors_m1_gt_middle{i}_V.txt', 0,
            f'{OUTPUT_MIDDLE_PLOT}/recalls_m1_middle{i}_V.png',
            f'{OUTPUT_MIDDLE_PLOT}/boxplot_m1_middle{i}_V')
        evaluate(
            f'{DATA_PATH}/cam_gt_m1.txt',
            f'{DATA_PATH}/middles/cam_eval_m1_middle{i}_H.txt',
            f'{OUTPUT_MIDDLE}/cam_errors_m1_gt_middle{i}_H.txt', 0,
            f'{OUTPUT_MIDDLE_PLOT}/recalls_m1_middle{i}_H.png',
            f'{OUTPUT_MIDDLE_PLOT}/boxplot_m1_middle{i}_H')

    # # batch process & compare
    # nr_batches = 4
    # for i in range(nr_batches):
    #     evaluate(
    #     f'{DATA_PATH}/cam_gt_m1.txt',
    #     f'{DATA_PATH}/Batch_init/model1/model1_init_batch{i}.txt',
    #     f'{OUTPUT_BATCH}/m1_init_batch{i}.txt', 0,
    #     f'{OUTPUT_BATCH_PLOT}/init_batch{i}_recalls_m1.png',
    #     f'{OUTPUT_BATCH_PLOT}/init_batch{i}_boxplot_m1')

    # for i in range(nr_batches):
    #     evaluate(
    #     f'{DATA_PATH}/cam_gt_m1.txt',
    #     f'{DATA_PATH}/Batch_final/model1/model1_final_batch{i}.txt',
    #     f'{OUTPUT_BATCH}/m1_final_batch{i}.txt', 0,
    #     f'{OUTPUT_BATCH_PLOT}/final_batch{i}_recalls_m1.png',
    #     f'{OUTPUT_BATCH_PLOT}/final_batch{i}_boxplot_m1')

    # # batch_entire_final
    for idx in range(1, nr_models):
        evaluate(
            f'{DATA_PATH}/cam_gt_m{idx}.txt',
            f'{DATA_PATH}/eval_m{idx}_batch_aligned.txt',
            f'{args.output_path}/batchalign_cam_errors_m{idx}_gt.txt', 0,
            f'{OUTPUT_PLOT}/batch_recalls_m{idx}.png',
            f'{OUTPUT_PLOT}/batch_boxplot_m{idx}')

    # # method comparison_ ICP
    for idx in range(1, nr_models):
        evaluate(
            f'{DATA_PATH}/cam_gt_m{idx}.txt',
            f'{DATA_PATH}/cam_eval_m{idx}_ICP.txt',
            f'{args.output_path}/ICP_cam_errors_m{idx}_gt.txt', 0,
            f'{OUTPUT_PLOT}/ICP_recalls_m{idx}.png',
            f'{OUTPUT_PLOT}/ICP_boxplot_m{idx}')

    # # method comparison_FGR
    for idx in range(1, nr_models):
        evaluate(
            f'{DATA_PATH}/cam_gt_m{idx}.txt',
            f'{DATA_PATH}/cam_eval_m{idx}_FGR.txt',
            f'{args.output_path}/FGR_cam_errors_m{idx}_gt.txt', 0,
            f'{OUTPUT_PLOT}/FGR_recalls_m{idx}.png',
            f'{OUTPUT_PLOT}/FGR_boxplot_m{idx}')

    # # method comparison_FRICP
    for idx in range(1, nr_models):
        evaluate(
            f'{DATA_PATH}/cam_gt_m{idx}.txt',
            f'{DATA_PATH}/fastICP_camera_extracted_m{idx}.txt', #_bd
            f'{args.output_path}/fastICP_cam_errors_m{idx}.txt', 0,
            f'{OUTPUT_PLOT}/fastICP_recalls_m{idx}.png',
            f'{OUTPUT_PLOT}/fastICP_boxplot_m{idx}')

    for idx in range(1, nr_models):
        evaluate(
            f'{DATA_PATH}/cam_gt_m{idx}.txt',
            f'{DATA_PATH}/robustICP_camera_extracted_m{idx}.txt', #_bd
            f'{args.output_path}/robustICP_bd_cam_errors_m{idx}.txt', 0,
            f'{OUTPUT_PLOT}/robustICP_bd_recalls_m{idx}.png',
            f'{OUTPUT_PLOT}/robustICP_bd_boxplot_m{idx}')

    # method comparison_TEASER
    for idx in range(1, nr_models):
        evaluate(
            f"{DATA_PATH}/cam_gt_m{idx}.txt",
            f"{DATA_PATH}/teaser_bd_camera_extracted_m{idx}.txt",
            f"{args.output_path}/Teaser_bd_cam_errors_m{idx}.txt",
            0,
            f"{OUTPUT_PLOT}/Teaser_bd_recalls_m{idx}.png",
            f"{OUTPUT_PLOT}/Teaser_bd_boxplot_m{idx}",
        )
