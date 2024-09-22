import pix4d_io as pix4d
from pathlib import Path

from subprocess import call, run

import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R
import shutil


# # --------------------------1st. (offset & image list) gt camera poses ------------------------------------
# def read_used_names(PATH_MODEL,i):
#     """
#     Read images that where used for sparse model
#     Returns:  ref_names(list of image names)
#     """
#     TXT_PATH = f'{PATH_MODEL}/model{i}_txt'
#     # Read image names from images.txt that where used for sparse model
#     images_df = pd.read_csv(f'{TXT_PATH}/images.txt', sep=' ', comment='#', header=None, usecols=range(10))
#     images_df.columns = ['IMAGE_ID', 'QW', 'QX', 'QY', 'QZ', 'TX', 'TY', 'TZ', 'CAMERA_ID', 'NAME']

#     images_df = images_df.drop(images_df.index[1::2])
#     ref_names = images_df['NAME'].tolist()

#     return ref_names

# # Function to read image names from a file into a list
# def read_image_list(image_list_path):
#     with open(image_list_path, 'r') as file:
#         return [line.strip() for line in file if line.strip()]

# # Function to read offset values from a file into a tuple
# def read_offset(offset_path):
#     with open(offset_path, 'r') as file:
#         offset = file.read().strip().split()
#         return tuple(float(value) for value in offset)

# # 3 diff views
# PROJECT_PATH = Path('3_disconnected_view/project_views')
# OUTPUT_PATH = Path(f'{PROJECT_PATH}/Correctness_loop/evaluation/gt_camera_pose')
# OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
# for i in range(2):
#     if i == 0: #summer_north
#         file_extern_cam_file = '3_disconnected_view/summer_north/apple-farm-20230627-north_calibrated_external_camera_parameters.txt'
#         IMAGE_LIST_PATH = '3_disconnected_view/summer_north/image_list_north.txt'
#         OFFSET_PATH = '3_disconnected_view/summer_north/apple-farm-20230627-north_offset.xyz'
#         output_filename = OUTPUT_PATH / 'gt_north_camera_poses.txt'
#     elif i == 1: #summer_east
#         file_extern_cam_file = '3_disconnected_view/summer_east/apple-farm-230209-east_calibrated_external_camera_parameters.txt' 
#         IMAGE_LIST_PATH = '3_disconnected_view/summer_east/image_list_east.txt'
#         # OFFSET_PATH = '2_different_season/summer/apple-farm-20230627-north_offset.xyz'
#         OFFSET_PATH = '3_disconnected_view/summer_north/apple-farm-20230627-north_offset.xyz'
#         output_filename = OUTPUT_PATH / 'gt_east_camera_poses.txt'

# # # 2cross seasons
# # PROJECT_PATH = Path('2_different_season/project_seasons')
# # OUTPUT_PATH = Path(f'{PROJECT_PATH}/Correctness_loop/evaluation/gt_camera_pose')
# # OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
# # for i in range(2):
# #     if i == 0: #winter
# #         file_extern_cam_file = '2_different_season/winter/applefarm-230209_merge_v9_calibrated_external_camera_parameters.txt'
# #         IMAGE_LIST_PATH = '2_different_season/winter/image_list_winter.txt'
# #         OFFSET_PATH = '2_different_season/winter/applefarm-230209_merge_v9_offset.xyz'
# #         output_filename = OUTPUT_PATH / 'gt_w_camera_poses.txt'
# #     elif i == 1: #summer
# #         file_extern_cam_file = '2_different_season/summer/apple-farm-20230627-north_calibrated_external_camera_parameters.txt' 
# #         IMAGE_LIST_PATH = '2_different_season/summer/image_list_summer.txt'
# #         # OFFSET_PATH = '2_different_season/summer/apple-farm-20230627-north_offset.xyz'
# #         OFFSET_PATH = '2_different_season/winter/applefarm-230209_merge_v9_offset.xyz'
# #         output_filename = OUTPUT_PATH / 'gt_sn_camera_poses.txt'

# # # 1neighboring
# # PROJECT_PATH = Path('try1/1_neighboring_4batches')
# # OUTPUT_PATH = Path(f'{PROJECT_PATH}/Correctness_loop/evaluation/gt_camera_pose')
# # OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
# # nr_batch = 4
# # for i in range(nr_batch):
# #     file_extern_cam_file = 'try1/applefarm-230209_merge_v9_calibrated_external_camera_parameters.txt'
# #     IMAGE_LIST_PATH = 'try1/image_list_neighboring.txt'
# #     OFFSET_PATH = 'try1/applefarm-230209_merge_v9_offset.xyz'
# #     output_filename = OUTPUT_PATH / f'gt_camera_poses_m{i}.txt'
    
    
    
#     image_names = read_image_list(IMAGE_LIST_PATH) 
#     # ALIGNED_MODEL_PATH = f'2_different_season/project_seasons/Correctness_loop/evaluation/aligned_model'
#     # ref_names = read_used_names(ALIGNED_MODEL_PATH,i)
#     INIT_ALIGNED_MODEL_PATH = f'{PROJECT_PATH}/Correctness_loop/init_ICP_swisstopo/Output/Init_transed_models'
#     ref_names = read_used_names(INIT_ALIGNED_MODEL_PATH,i)
#     offset = read_offset(OFFSET_PATH)
#     #cx = 2678199.0 
#     #cy = 1258409.0
#     #cz = 445.000
#     # print('offset',offset)

#     points, orientations, filenames = pix4d.load_pix4d_sfm_model(file_extern_cam_file, ref_names, offset)

#     with open(output_filename, 'w') as f:
#         f.write('filename, x, y, z, qw, qx, qy, qz\n')
#         for filename, point, orientation in zip(filenames, points, orientations):
#             f.write(f'{filename}, {point[0]}, {point[1]}, {point[2]}, {orientation[0]}, {orientation[1]}, {orientation[2]}, {orientation[3]}\n')
#     print(f'model{i} done!')

#     CAM_COMPARE_PATH = Path(f'{PROJECT_PATH}/Correctness_loop/evaluation/Camera_poses_compare')
#     CAM_COMPARE_PATH.mkdir(parents=True, exist_ok=True)
#     cam_compare_file_gt = f'{CAM_COMPARE_PATH}/cam_gt_m{i}.txt'
#     with open(cam_compare_file_gt, 'w') as f:
#         f.write('filename qw qx qy qz x y z\n')
#         for filename, point, orientation in zip(filenames, points, orientations):
#             f.write(f'{filename} {orientation[0]} {orientation[1]} {orientation[2]} {orientation[3]} {point[0]} {point[1]} {point[2]}\n')



# # # -------------------------2nd. colmap model to .txt----------------------------
# # # can also be done using colmap model_converter: run in terminal
# # Convert model to txt format
# TXT_PATH = Path(f'2_different_season/project_seasons/Correctness_loop/init_ICP_swisstopo/Output/Init_transed_models/cropped_model_0/model_segment_0_2_txt')
# TXT_PATH.mkdir(parents=True, exist_ok=True)
# model_converter_command = [
# 'colmap',
# 'model_converter',
# '--input_path', f'2_different_season/project_seasons/Correctness_loop/init_ICP_swisstopo/Output/Init_transed_models/cropped_model_0/model_segment_0_2',
# '--output_path', f'{TXT_PATH}',
# '--output_type', 'TXT'
# ]
# call(model_converter_command)

# # # using custom script 
# # PROJECT_PATH = '2_different_season/project_seasons'
# # # INPUT_MODEL_PATH = f'{PROJECT_PATH}/Models/Model_0/shifted_LV95_model'
# # INPUT_MODEL_PATH = f'{PROJECT_PATH}/Models/Model_0/sparse/0'
# # # OUTPUT_MODEL_TXT_PATH = f'{PROJECT_PATH}/Models/Model_0/shifted_test_txt'
# # OUTPUT_MODEL_TXT_PATH = f'{PROJECT_PATH}/Models/Model_0/sparse_test_txt/custom_file'
# # create_image_batch_model0_command = [
# #     'python3',
# #     'Apple-Farm/evaluation/read_write_model.py',
# #     "--input_model", f'{INPUT_MODEL_PATH}',
# #     "--input_format", ".bin",
# #     "--output_model", f'{OUTPUT_MODEL_TXT_PATH}',
# #     "--output_format", ".txt", 
# #     ]
# # call(create_image_batch_model0_command)



# # --- 3/2.2 extract the image info(camera poses from shifted model) in the same way as gt ---
# nr_batch = 2
# # for idx in range(nr_batch):
# for idx in range(1,2):
#     # TXT_PATH = f'2_different_season/project_seasons/Models/Model_{idx}/shifted_LV95_model_txt'
#     # TXT_PATH = f'2_different_season/project_seasons/Correctness_loop/evaluation/aligned_model/model{idx}_txt'
#     # TXT_PATH = f'2_different_season/project_seasons/Correctness_loop/3_batch_align/Init_files/Models/model{idx}_txt'

#     # TXT_PATH = f'2_different_season/project_seasons/Correctness_loop/evaluation/gt_models/GT_Model_{idx}_txt'
#     TXT_PATH = f'2_different_season/project_seasons/Correctness_loop/evaluation/gt_models/Eval_Model_1_txt' # m1 alone
#     file_name = 'images.txt'
#     images_df = pd.read_csv(f'{TXT_PATH}/{file_name}', sep=' ', comment='#', header=None, usecols=range(10))
#     images_df.columns = ['IMAGE_ID', 'QW', 'QX', 'QY', 'QZ', 'TX', 'TY', 'TZ', 'CAMERA_ID', 'NAME']

#     images_df = images_df.drop(images_df.index[1::2]).reset_index(drop=True)
#     # extracted_df = images_df[['IMAGE_ID', 'QW', 'QX', 'QY', 'QZ', 'TX', 'TY', 'TZ', 'NAME']]
#     # Compute the camera center for each image
#     camera_centers = []
#     for index, row in images_df.iterrows():
#         q = [row['QX'], row['QY'], row['QZ'], row['QW']]
#         t = np.array([row['TX'], row['TY'], row['TZ']])
        
#         # Convert quaternion to rotation matrix
#         rotation = R.from_quat(q)
#         rotation_matrix = rotation.as_matrix()
        
#         # Calculate camera center in world coordinates
#         camera_center = -rotation_matrix.T @ t
#         camera_centers.append(camera_center)
#     camera_centers_df = pd.DataFrame(camera_centers, columns=['CX', 'CY', 'CZ'])
#     output = pd.concat([images_df['NAME'],camera_centers_df],axis=1)

#     # OUTPUT_PATH = Path(f'2_different_season/project_seasons/Models/Model_{idx}/shifted_LV95_model_txt/camera_pose')
#     # OUTPUT_PATH = Path(f'2_different_season/project_seasons/Correctness_loop/evaluation/aligned_model/model{idx}_txt')
#     # OUTPUT_PATH.mkdir(parents=True,exist_ok=True)
#     # extracted_df.to_csv(f'{OUTPUT_PATH}/extracted_trans.txt', sep=' ', index=False, header=True)

#     OUTPUT_PATH = TXT_PATH
#     output.to_csv(f'{OUTPUT_PATH}/extracted_cam{idx}.txt', sep=' ', index=False, header=True)

#     # for camera poses compare
#     output_poses = pd.concat([images_df['NAME'],images_df['QW'],images_df['QX'],images_df['QY'],images_df['QZ'],camera_centers_df],axis=1)
#     header = ["filename", "qw", "qx", "qy", "qz", "x", "y", "z"]
#     CAM_COMPARE_PATH = Path('2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare')
#     CAM_COMPARE_PATH.mkdir(parents=True, exist_ok=True)
#     # output_poses.to_csv(f'{CAM_COMPARE_PATH}/cam_gtextract_m{idx}.txt', sep=' ', index=False, header=header)
#     output_poses.to_csv(f'{CAM_COMPARE_PATH}/cam_eval_m{idx}.txt', sep=' ', index=False, header=header)  # m1 alone


# # # ------------------------- 3rd plot box-plot ----------------------
import matplotlib.pyplot as plt
import os
from matplotlib.gridspec import GridSpec

# def plot_comparison_boxplot(errors_list, labels, output_file, y_lim=None, whiskers=[25, 75], title_suffix=''):
#     plt.figure(figsize=(10, 6))
#     boxplot_objects = plt.boxplot(errors_list, showfliers=True, whis=whiskers, labels=labels)
    
#     if y_lim:
#         plt.ylim(y_lim)

#     # Annotate the statistics
#     for i, error in enumerate(errors_list):
#         stats = boxplot_objects['whiskers'][i*2].get_ydata()
#         Q1, Q3 = np.percentile(error, [25, 75])
#         median = np.median(error)
#         whisker_low, whisker_high = np.percentile(error, whiskers)
#         plt.text(i+1.1, median, f'Median: {median:.2f}', verticalalignment='center')
#         plt.text(i+1.1, Q1, f'25th: {Q1:.2f}', verticalalignment='center')
#         plt.text(i+1.1, Q3, f'75th: {Q3:.2f}', verticalalignment='center')
#         # plt.text(i+1.1, whisker_low, f'{whiskers[0]}th: {whisker_low:.2f}', verticalalignment='center')
#         # plt.text(i+1.1, whisker_high, f'{whiskers[1]}th: {whisker_high:.2f}', verticalalignment='center')

#     plt.title(f'Comparison of {title_suffix} Errors')
#     plt.ylabel('Error Values')
#     plt.savefig(output_file)
#     plt.show()

# def plot_comparison_boxplot(errors_list, labels, output_file, y_lim=None, whiskers=[25, 75], title_suffix=''):
#     # Calculate global min and max based on IQR
#     iqr_limits = []
#     for error in errors_list:
#         Q1, Q3 = np.percentile(error, [10, 90])
#         IQR = Q3 - Q1
#         iqr_limits.append((Q1 - 0.5 * IQR, Q3 + 0.5 * IQR))

#     # Determine which error sets need individual subplots
#     needs_subplot = [False] * len(errors_list)
#     for i in range(len(errors_list)):
#         if i > 0:
#             prev_diff = iqr_limits[i][1] - iqr_limits[i - 1][1]
#             needs_subplot[i] = needs_subplot[i] or prev_diff > 10
#         if i < len(errors_list) - 1:
#             next_diff = iqr_limits[i][1] - iqr_limits[i + 1][1]
#             needs_subplot[i] = needs_subplot[i] or next_diff > 10

#     global_max = max(limit[1] for limit in iqr_limits)
#     global_ylim_min = min(limit[0] for limit in iqr_limits)

#     # if global_max - global_min < 10:
#     if all(not need for need in needs_subplot):
#         plt.figure(figsize=(10, 6))
#         boxplot_objects = plt.boxplot(errors_list, showfliers=False, whis=whiskers, labels=labels)

#         plt.ylim([global_ylim_min-1, global_max])

#         for i, error in enumerate(errors_list):
#             stats = boxplot_objects['whiskers'][i*2].get_ydata()
#             Q1, Q3 = np.percentile(error, [25, 75])
#             median = np.median(error)
#             whisker_low, whisker_high = np.percentile(error, whiskers)
#             # plt.text(i+1.1, median, f'Median: {median:.2f}', verticalalignment='center')
#             # plt.text(i+1.1, Q1, f'25th: {Q1:.2f}', verticalalignment='center')
#             # plt.text(i+1.1, Q3, f'75th: {Q3:.2f}', verticalalignment='center')
            
#             # Determine vertical positions for text to avoid overlap
#             text_positions = {median: median, Q1: Q1, Q3: Q3}
#             min_distance = max(0.02 * (global_max-global_ylim_min), 0.08)#* (whisker_high - whisker_low) 

#             if median - Q1 < min_distance:
#                 Q1_text_pos = Q1 - min_distance
#             else:
#                 Q1_text_pos = Q1

#             if Q3 - median < min_distance:
#                 Q3_text_pos = Q3 + min_distance
#             else:
#                 Q3_text_pos = Q3

#             plt.text(i+1.1, median, f'Median: {median:.2f}', verticalalignment='center')
#             plt.text(i+1.1, Q1_text_pos, f'25th: {Q1:.2f}', verticalalignment='center')
#             plt.text(i+1.1, Q3_text_pos, f'75th: {Q3:.2f}', verticalalignment='center')


#         plt.title(f'{title_suffix}')
#         plt.ylabel('Error Values')
#         plt.savefig(output_file)
#         plt.show()
#     else:
#         unique_ylim_min = min(iqr_limits[i][0] for i, need in enumerate(needs_subplot) if not need)
#         unique_ylim_max = max(iqr_limits[i][1] for i, need in enumerate(needs_subplot) if not need) # case1 + 0.1

#         fig, axs = plt.subplots(1, len(errors_list), figsize=(4* len(errors_list), 4))
        
#         for i, error in enumerate(errors_list):
#             ax = axs[i] if len(errors_list) > 1 else axs
#             ax.boxplot(error, showfliers=True, whis=whiskers)
#             ax.set_xlabel(f'{labels[i]}')
#             # ax.set_ylabel('Error Values')
            
#             Q1, median, Q3 = np.percentile(error, [25, 50, 75])
#             min_distance = max(0.01 * (Q3 - Q1), 0.1)

#             Q1_text_pos = Q1 - min_distance if median - Q1 < min_distance else Q1
#             Q3_text_pos = Q3 + min_distance if Q3 - median < min_distance else Q3

#             ax.text(1.1, median, f'Median: {median:.2f}', verticalalignment='center')
#             ax.text(1.1, Q1_text_pos, f'25th: {Q1:.2f}', verticalalignment='center')
#             ax.text(1.1, Q3_text_pos, f'75th: {Q3:.2f}', verticalalignment='center')

#             ax.set_xticklabels([])
#             if needs_subplot[i]:
#                 Q1, Q3 = np.percentile(error, [10, 90])
#                 IQR = Q3 - Q1
#                 ax.set_ylim([Q1 - 0.5 * IQR, Q3 + 0.5 * IQR])
#             else:
#                 ax.set_ylim([unique_ylim_min, unique_ylim_max])

#         axs[0].set_ylabel('Error Values')
#         # plt.tight_layout()
#         plt.suptitle(f'{title_suffix}')
#         plt.savefig(output_file)
#         plt.show()

def boxplot_size_change(box):
    # Customize boxplot elements
    for median in box['medians']:
        median.set_color('r')
        median.set_linewidth(6)  # Make the median line bolder

    for whisker in box['whiskers']:
        whisker.set_linewidth(4)  # Change the whiskers' width if desired

    for cap in box['caps']:
        cap.set_linewidth(4)  # Change the caps' width if desired

    for box in box['boxes']:
        box.set_linewidth(4)  # Change the boxes' width if desired




def plot_comparison_boxplot(errors_list, labels, output_file, y_lim=None, whiskers=[25, 75], title_suffix='', yaxis = ''):
    # Calculate global min and max based on IQR
    iqr_limits = []
    for error in errors_list:
        Q1, Q3 = np.percentile(error, [10, 90])
        IQR = Q3 - Q1
        iqr_limits.append((Q1 - 0.5 * IQR, Q3 + 0.5 * IQR))

    # # Check if only the first error set needs an individual subplot
    diff = 6 # if global_max - global_min < 10:
    separate_first = False
    if len(errors_list) > 1:
        first_diff_next = iqr_limits[0][1] - iqr_limits[1][1]
        separate_first = abs(first_diff_next) > diff
    separate_first = True

    flierprops = dict(marker='o', markerfacecolor='none', markersize=10, linestyle='none', markeredgecolor='gray')
    if not separate_first:
        global_max = max(limit[1] for limit in iqr_limits)
        global_ylim_min = min(limit[0] for limit in iqr_limits)

        plt.figure(figsize=(18, 10))
        boxplot_objects = plt.boxplot(errors_list, showfliers=True, whis=whiskers, labels=labels, widths=0.6, flierprops=flierprops)
        boxplot_size_change(boxplot_objects)

        plt.ylim([global_ylim_min + 0.1, global_max])
        plt.yticks(fontsize=26)
        plt.xticks(fontsize=24) #30
        plt.title(f'{title_suffix}',fontsize=38)
        plt.ylabel(f'{yaxis}',fontsize=30)
        plt.savefig(output_file)
        plt.show()
    else:
        unique_ylim_min = min(iqr_limits[i][0] for i in range(1,len(iqr_limits)))
        unique_ylim_max = max(iqr_limits[i][1] for i in range(1,len(iqr_limits))) # case1 + 0.1

        plt.figure(figsize=(18, 7.5))
        gs = GridSpec(1, 2, width_ratios=[1,4])  #  3,5
        plt.subplots_adjust(wspace=0.2)  # 0.4
 
        # Subplot for the first error set
        ax1 = plt.subplot(gs[0])
        box = ax1.boxplot(errors_list[0], showfliers=True, whis=whiskers, labels=[labels[0]], widths=0.6, flierprops=flierprops)
        boxplot_size_change(box)
        ax1.set_ylim([iqr_limits[0][0]  , iqr_limits[0][1]]) # -1 , 0  # 0,0
        ax1.set_ylabel(yaxis, fontsize=38)
        plt.yticks(fontsize=34)
        # plt.xticks(fontsize=32)
        ax1.tick_params(axis='x', labelsize=32)
        

        # Subplot for the remaining error sets
        ax2 = plt.subplot(gs[1])
        box = ax2.boxplot(errors_list[1:], showfliers=True, whis=whiskers, labels=labels[1:], widths=0.7, flierprops=flierprops)
        boxplot_size_change(box)
        ax2.set_ylim([unique_ylim_min+0.1 , unique_ylim_max ]) # + 0.1 , - 0.2 #+0.2,0
        # ax2.set_ylabel(yaxis, fontsize=33)
        plt.yticks(fontsize=34)
        # plt.xticks(fontsize=32)
        ax2.tick_params(axis='x', labelsize=32) 

        plt.subplots_adjust(bottom=0.15)
        plt.suptitle(f'{title_suffix}',fontsize=38)
        plt.savefig(output_file)
        plt.show()



# # ------------------ init-sarah-mein --------------------
# #1 case
# nr_models = 4
# for idx in range(1,2):
#     colmap_init_file_t = f'try1/1_neighboring_4batches/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/co_init_boxplot_m{idx}_t.txt'
#     sarah_error_file_t = f'try1/1_neighboring_4batches/Matched_models/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/sarah_aligned_boxplot_m{idx}_t.txt'
#     reconstruction_error_file_t = f'try1/1_neighboring_4batches/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/reconstruction_boxplot_m{idx}_t.txt'
#     batch_error_file_t = f'try1/1_neighboring_4batches/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/batch_boxplot_m{idx}_t.txt'
#     loop_error_file_t = f'try1/1_neighboring_4batches/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/loop_boxplot_m{idx}_t.txt'
#     swisstopo_error_file_t = f'try1/1_neighboring_4batches/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/swisstopo_boxplot_m{idx}_t.txt'

#     colmap_init_file_r = f'try1/1_neighboring_4batches/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/co_init_boxplot_m{idx}_r.txt'
#     sarah_error_file_r = f'try1/1_neighboring_4batches/Matched_models/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/sarah_aligned_boxplot_m{idx}_r.txt'
#     reconstruction_error_file_r = f'try1/1_neighboring_4batches/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/reconstruction_boxplot_m{idx}_r.txt'
#     batch_error_file_r = f'try1/1_neighboring_4batches/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/batch_boxplot_m{idx}_r.txt'
#     loop_error_file_r = f'try1/1_neighboring_4batches/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/loop_boxplot_m{idx}_r.txt'
#     swisstopo_error_file_r = f'try1/1_neighboring_4batches/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/swisstopo_boxplot_m{idx}_r.txt'

# # 2case
# nr_models = 2
# for idx in range(1,nr_models): 
#     colmap_init_file_t =  f'2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/co_init_boxplot_m1_t.txt'
#     reconstruction_error_file_t = f'2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/reconstruction_boxplot_m1_t.txt'
#     batch_error_file_t = f'2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/batch_boxplot_m1_t.txt'
#     loop_error_file_t = f'2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/loop_boxplot_m1_t.txt'
#     swisstopo_error_file_t = f'2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/swisstopo_boxplot_m1_t.txt'
#     # sarah_error_file_t = f'2_different_season/project_sarah/Matched_models/evaluation_baseline/Camera_poses_compare/Cam_pose_Error/Figures/sarah_aligned_boxplot_m1_t.txt'

#     colmap_init_file_r =  f'2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/co_init_boxplot_m1_r.txt'
#     reconstruction_error_file_r = f'2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/reconstruction_boxplot_m1_r.txt'
#     batch_error_file_r = f'2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/batch_boxplot_m1_r.txt'
#     loop_error_file_r = f'2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/loop_boxplot_m1_r.txt'
#     swisstopo_error_file_r = f'2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/swisstopo_boxplot_m1_r.txt'
#     # sarah_error_file_r = f'2_different_season/project_sarah/Matched_models/evaluation_baseline/Camera_poses_compare/Cam_pose_Error/Figures/sarah_aligned_boxplot_m1_r.txt'

# #3 case
# nr_models = 2
# for idx in range(1,nr_models):
#     colmap_init_file_t = f'3_disconnected_view/project_views/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/co_init_boxplot_m1_t.txt' 
#     reconstruction_error_file_t = f'3_disconnected_view/project_views/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/reconstruction_boxplot_m1_t.txt'
#     batch_error_file_t = f'3_disconnected_view/project_views/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/batch_boxplot_m1_t.txt'
#     sarah_error_file_t = f'3_disconnected_view/project_sarah/Matched_models/evaluation_baseline/Camera_poses_compare/Cam_pose_Error/Figures/sarah_aligned_boxplot_m1_t.txt'
#     loop_error_file_t = f'3_disconnected_view/project_views/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/loop_boxplot_m1_t.txt'
#     swisstopo_error_file_t = f'3_disconnected_view/project_views/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/swisstopo_boxplot_m1_t.txt'

#     colmap_init_file_r = f'3_disconnected_view/project_views/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/co_init_boxplot_m1_r.txt' 
#     reconstruction_error_file_r = f'3_disconnected_view/project_views/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/reconstruction_boxplot_m1_r.txt'
#     batch_error_file_r = f'3_disconnected_view/project_views/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/batch_boxplot_m1_r.txt'
#     sarah_error_file_r = f'3_disconnected_view/project_sarah/Matched_models/evaluation_baseline/Camera_poses_compare/Cam_pose_Error/Figures/sarah_aligned_boxplot_m1_r.txt'
#     loop_error_file_r = f'3_disconnected_view/project_views/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/loop_boxplot_m1_r.txt'
#     swisstopo_error_file_r = f'3_disconnected_view/project_views/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/swisstopo_boxplot_m1_r.txt'

# #  andreas zigzag
# nr_models = 3
# Project_path = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/project_new_andreas_zigzagpair'
# Error_path = f'{Project_path}/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures'

# for idx in range(1,nr_models): 
#     colmap_init_file_t =  f'{Error_path}/co_init_boxplot_m{idx}_t.txt'
#     reconstruction_error_file_t = f'{Error_path}/reconstruction_boxplot_m{idx}_t.txt'
#     batch_error_file_t = f'{Error_path}/batch_boxplot_m{idx}_t.txt'
#     loop_error_file_t = f'{Error_path}/loop_boxplot_m{idx}_t.txt'
#     swisstopo_error_file_t = f'{Error_path}/initial_aligned_boxplot_m{idx}_t.txt'

#     colmap_init_file_r =  f'{Error_path}/co_init_boxplot_m{idx}_r.txt'
#     reconstruction_error_file_r = f'{Error_path}/reconstruction_boxplot_m{idx}_r.txt'
#     batch_error_file_r = f'{Error_path}/batch_boxplot_m{idx}_r.txt'
#     loop_error_file_r = f'{Error_path}/loop_boxplot_m{idx}_r.txt'
#     swisstopo_error_file_r = f'{Error_path}/initial_aligned_boxplot_m{idx}_r.txt'

       
# # andreas predicted
# nr_models = 3
# Project_path = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/project_zigzag_original'

# diff views 
nr_models = 3
Project_path = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case1/project_views'

# # diff seasons
# nr_models = 2
# Project_path = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case2/project_seasons'

# -----------------------
Error_path = f'{Project_path}/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures'

for idx in range(2,nr_models): 
    colmap_init_file_t =  f'{Error_path}/co_init_boxplot_m{idx}_t.txt'
    reconstruction_error_file_t = f'{Error_path}/reconstruction_boxplot_m{idx}_t.txt'
    batch_error_file_t = f'{Error_path}/batch_boxplot_m{idx}_t.txt'
    loop_error_file_t = f'{Error_path}/loop_boxplot_m{idx}_t.txt'
    swisstopo_error_file_t = f'{Error_path}/initial_aligned_boxplot_m{idx}_t.txt'

    colmap_init_file_r =  f'{Error_path}/co_init_boxplot_m{idx}_r.txt'
    reconstruction_error_file_r = f'{Error_path}/reconstruction_boxplot_m{idx}_r.txt'
    batch_error_file_r = f'{Error_path}/batch_boxplot_m{idx}_r.txt'
    loop_error_file_r = f'{Error_path}/loop_boxplot_m{idx}_r.txt'
    swisstopo_error_file_r = f'{Error_path}/initial_aligned_boxplot_m{idx}_r.txt'
   

# # camparison methods
# for idx in range(1,2):      
#     PROJECT_PATH = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case2/project_seasons'  
#     # PROJECT_PATH =f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/project_new_andreas_zigzagpair'
#     error_path = f'{PROJECT_PATH}/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures'
#     ours_error_file_t = f'{error_path}/batch_boxplot_m{idx}_t.txt'
#     compare_icp_file_t = f'{error_path}/ICP_boxplot_m{idx}_t.txt'
#     compare_fgr_file_t = f'{error_path}/FGR_boxplot_m{idx}_t.txt'
#     compare_ficp_bd_file_t = f'{error_path}/fastICP_boxplot_m{idx}_t.txt'
#     compare_ricp_bd_file_t = f'{error_path}/robustICP_bd_boxplot_m{idx}_t.txt'
#     compare_teaser_bd_file_t = f'{error_path}/Teaser_bd_boxplot_m{idx}_t.txt'

#     ours_error_file_r = f'{error_path}/batch_boxplot_m{idx}_r.txt'
#     compare_icp_file_r = f'{error_path}/ICP_boxplot_m{idx}_r.txt'
#     compare_fgr_file_r = f'{error_path}/FGR_boxplot_m{idx}_r.txt'
#     compare_ficp_bd_file_r = f'{error_path}/fastICP_boxplot_m{idx}_r.txt'
#     compare_ricp_bd_file_r = f'{error_path}/robustICP_bd_boxplot_m{idx}_r.txt'
#     compare_teaser_bd_file_r = f'{error_path}/Teaser_bd_boxplot_m{idx}_r.txt'

# ------------------------------
    # Read the errors from the files
    err_co_init_t = np.loadtxt(colmap_init_file_t)
    err_reconstruction_t = np.loadtxt(reconstruction_error_file_t)
    err_batch_t = np.loadtxt(batch_error_file_t)
    err_loop_t = np.loadtxt(loop_error_file_t)
    # err_sarah_t = np.loadtxt(sarah_error_file_t)
    err_swiss_t = np.loadtxt(swisstopo_error_file_t)

    err_reconstruction_r = np.loadtxt(reconstruction_error_file_r)
    err_batch_r = np.loadtxt(batch_error_file_r)
    err_co_init_r = np.loadtxt(colmap_init_file_r)
    err_loop_r = np.loadtxt(loop_error_file_r)
    err_swiss_r = np.loadtxt(swisstopo_error_file_r)
    # err_sarah_r = np.loadtxt(sarah_error_file_r)
    
    # # compare
    # err_ours_t = np.loadtxt(ours_error_file_t)
    # err_ICP_t = np.loadtxt(compare_icp_file_t)
    # err_FGR_t = np.loadtxt(compare_fgr_file_t)
    # err_FICP_bd_t = np.loadtxt(compare_ficp_bd_file_t)
    # err_RICP_bd_t = np.loadtxt(compare_ricp_bd_file_t)
    # err_teaser_bd_t = np.loadtxt(compare_teaser_bd_file_t)

    # err_ours_r = np.loadtxt(ours_error_file_r)
    # err_ICP_r = np.loadtxt(compare_icp_file_r)
    # err_FGR_r = np.loadtxt(compare_fgr_file_r)
    # err_FICP_bd_r = np.loadtxt(compare_ficp_bd_file_r)
    # err_RICP_bd_r = np.loadtxt(compare_ricp_bd_file_r)
    # err_teaser_bd_r = np.loadtxt(compare_teaser_bd_file_r)
# -----------------------------------
    # # Define the labels for the plots
    # labels_t = ['Initial', 'Baseline', 'Current Method']
    # labels_r = ['Initial', 'Baseline', 'Current Method']
    # output_boxplot_T = f'vidual_pre/EVAL/cam/case1/refined_plot/refined_init_sarah_mein_t{idx}.png'
    # output_boxplot_R = f'vidual_pre/EVAL/cam/case1/refined_plot/refined_init_sarah_mein_r{idx}.png'

    labels_t = ['Colmap\ninitial', 'Initial\naligned', 'Loop\naligned', 'Batch\naligned','LB']
    labels_r = ['Colmap\ninitial', 'Initial\naligned', 'Loop\naligned', 'Batch\naligned','LB']
    output_boxplot_T = f'{Error_path}/ours_steps_t{idx}.svg' #png
    output_boxplot_R = f'{Error_path}/ours_steps_r{idx}.svg'
    
    # labels_t = ['Ours', 'ICP', 'FGR', 'FastICP', 'RobustICP', 'Teaser++'] #, 'Teaser++'
    # labels_r = ['Ours', 'ICP', 'FGR', 'FastICP', 'RobustICP', 'Teaser++']
    # output_boxplot_T = f'{error_path}/compare_methods/compare_methods_t{idx}.png'
    # output_boxplot_R = f'{error_path}/compare_methods/compare_methods_r{idx}.png'
    
    y_lim_t = True #True #[0,5] #None  #3: 0.6  #2: 1.0
    y_lim_r = True #True #[0,5] #None  #3: 0.6  #2: 1.0
    whiskers_t = [10,90]
    whiskers_r = [10,90]

    # plot_comparison_boxplot([err_co_init_t, err_sarah_t, err_batch_t], labels_t, output_boxplot_T, y_lim_t, whiskers_t, title_suffix=f'Comparison of Translation Error (model{idx})',yaxis=f'Error Value (m)') #(err_t, output_boxplot_T, y_lim_t, whiskers_t)
    # plot_comparison_boxplot([err_co_init_r, err_sarah_r, err_batch_r], labels_r, output_boxplot_R, y_lim_r, whiskers_r, title_suffix=f'Comparison of Rotation Error (model{idx})',yaxis=f'Error Value (degree)') #(err_r, output_boxplot_R, y_lim_r, whiskers_r)
    
    # plot_comparison_boxplot([err_co_init_t, err_swiss_t, err_loop_t, err_batch_t, err_reconstruction_t], labels_t, output_boxplot_T, y_lim_t, whiskers_t, title_suffix=f'Comparison of Translation Error (model_north view)',yaxis=f'Error Value (m)') #(err_t, output_boxplot_T, y_lim_t, whiskers_t)
    plot_comparison_boxplot([err_co_init_r, err_swiss_r, err_loop_r, err_batch_r, err_reconstruction_r], labels_r, output_boxplot_R, y_lim_r, whiskers_r, title_suffix=f'Comparison of Rotation Error (model_north view)',yaxis=f'Error Value (degree)') #(err_r, output_boxplot_R, y_lim_r, whiskers_r)

#     plot_comparison_boxplot([err_ours_t, err_ICP_t, err_FGR_t, err_FICP_bd_t, err_RICP_bd_t, err_teaser_bd_t], labels_t, output_boxplot_T, y_lim_t, whiskers_t, title_suffix=f'Comparison of Translation Error (model_{idx})',yaxis=f'Error Value (m)') 
#     plot_comparison_boxplot([err_ours_r, err_ICP_r, err_FGR_r, err_FICP_bd_r, err_RICP_bd_r, err_teaser_bd_r], labels_r, output_boxplot_R, y_lim_r, whiskers_r, title_suffix=f'Comparison of Rotation Error (model_{idx})',yaxis=f'Error Value (degree)')  
# #, err_teaser_bd_t, err_teaser_bd_r
    




# # ---------------------- base; loop; batch ---------------------------------
# # # 2case
# # reconstruction_error_file_t = f'2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/reconstruction_boxplot_m1_t.txt'
# # batch_error_file_t = f'2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/batch_boxplot_m1_t.txt'
# # loop_error_file_t = f'2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/loop_boxplot_m1_t.txt'

# # reconstruction_error_file_r = f'2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/reconstruction_boxplot_m1_r.txt'
# # batch_error_file_r = f'2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/batch_boxplot_m1_r.txt'
# # loop_error_file_r = f'2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/loop_boxplot_m1_r.txt'

# # #3 case
# # reconstruction_error_file_t = f'3_disconnected_view/project_views/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/reconstruction_boxplot_m1_t.txt'
# # batch_error_file_t = f'3_disconnected_view/project_views/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/batch_boxplot_m1_t.txt'
# # loop_error_file_t = f'3_disconnected_view/project_views/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/loop_boxplot_m1_t.txt'

# # reconstruction_error_file_r = f'3_disconnected_view/project_views/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/reconstruction_boxplot_m1_r.txt'
# # batch_error_file_r = f'3_disconnected_view/project_views/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/batch_boxplot_m1_r.txt'
# # loop_error_file_r = f'3_disconnected_view/project_views/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/loop_boxplot_m1_r.txt'

# #1 case
# reconstruction_error_file_t = f'try1/1_neighboring_4batches/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/reconstruction_boxplot_m3_t.txt'
# batch_error_file_t = f'try1/1_neighboring_4batches/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/batch_boxplot_m3_t.txt'
# loop_error_file_t = f'try1/1_neighboring_4batches/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/loop_boxplot_m3_t.txt'

# reconstruction_error_file_r = f'try1/1_neighboring_4batches/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/reconstruction_boxplot_m3_r.txt'
# batch_error_file_r = f'try1/1_neighboring_4batches/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/batch_boxplot_m3_r.txt'
# loop_error_file_r = f'try1/1_neighboring_4batches/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/loop_boxplot_m3_r.txt'

# # Read the errors from the files
# err_reconstruction_t = np.loadtxt(reconstruction_error_file_t)
# err_batch_t = np.loadtxt(batch_error_file_t)
# err_reconstruction_r = np.loadtxt(reconstruction_error_file_r)
# err_batch_r = np.loadtxt(batch_error_file_r)
# err_loop_t = np.loadtxt(loop_error_file_t)
# err_loop_r = np.loadtxt(loop_error_file_r)
# # Define the labels for the plots
# labels_t = ['Reconstruction', 'Loop', 'Batch']
# labels_r = ['Reconstruction', 'Loop', 'Batch']

# output_boxplot_T = f'try1/1_neighboring_4batches/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/base_loop_batch_t3.png'
# output_boxplot_R = f'try1/1_neighboring_4batches/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/base_loop_batch_r3.png'
# y_lim_t = [0,0.6] #True #[0,5] #None  #3: 0.6  #2: 1.0
# y_lim_r = [0,1.2] #True #[0,5] #None  #3: 0.6  #2: 1.0
# whiskers_t = [10,90]
# whiskers_r = [10,90]

# plot_comparison_boxplot([err_reconstruction_t, err_loop_t, err_batch_t], labels_t, output_boxplot_T, y_lim_t, whiskers_t, title_suffix='Translation') #(err_t, output_boxplot_T, y_lim_t, whiskers_t)
# plot_comparison_boxplot([err_reconstruction_r, err_loop_r, err_batch_r], labels_r, output_boxplot_R, y_lim_r, whiskers_r, title_suffix='Rotation') #(err_r, output_boxplot_R, y_lim_r, whiskers_r)


# -------
# # batches
# error_files_directory = '2_different_season/project_seasons/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures/batches_plot'

# errors_translation = []
# errors_rotation = []
# labels = []

# # Loop through the file indices and read errors from each file
# for i in range(5):
#     translation_file = os.path.join(error_files_directory, f'final_batch{i}_boxplot_m1_t.txt')
#     rotation_file = os.path.join(error_files_directory, f'final_batch{i}_boxplot_m1_r.txt')
    
#     # Read the error values from the files (replace with actual file reading in the real scenario)
#     errors_translation.append(np.loadtxt(translation_file))
#     errors_rotation.append(np.loadtxt(rotation_file))
    
#     # Add labels for the plots
#     labels.append(f'Batch {i}')
# y_lim = [0, 1.75] 

# output_file_translation = os.path.join(error_files_directory, 'comparison_final_errors_t.png')
# output_file_rotation = os.path.join(error_files_directory, 'comparison_final_errors_r.png')

# plot_comparison_boxplot(errors_translation, labels, output_file_translation, y_lim, title_suffix='Translation')
# plot_comparison_boxplot(errors_rotation, labels, output_file_rotation, y_lim, title_suffix='Rotation')