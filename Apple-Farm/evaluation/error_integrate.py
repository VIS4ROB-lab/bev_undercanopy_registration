import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# camer_error_path = '/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/project_new_andreas_zigzagpair/Correctness_loop/evaluation/Camera_poses_compare/Cam_pose_Error/Figures'
# output_path = Path(f'{camer_error_path}/integrate_model12')
# output_path.mkdir(parents=True, exist_ok=True)
# # batch_boxplot_m2_t


# # method_name = ['batch','ICP','FGR','fastICP','robustICP_bd','Teaser_bd'] # _boxplot_m{idx}_t.txt
# # method_name = ['Teaser_bd']
# method_name = ['batch']
# err_t_tosave = {}
# for mn in method_name:
#     err_t_all = []
#     for idx in range(1,3):
#         camer_error_t_file = f'{camer_error_path}/{mn}_boxplot_m{idx}_r.txt'
#         err_t = np.loadtxt(camer_error_t_file)
#         err_t_all = np.concatenate((err_t_all, err_t), axis=None)
#     err_t_tosave[mn] = np.median(err_t_all)
# np.savetxt(f'{output_path}/integrated_cam_error_r.txt',list(err_t_tosave.items()), fmt='%s', delimiter=",", header='method,median_error')


# ------------ tree errors integration -------------
tree_error_path = '/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/project_new_andreas_zigzagpair/Correctness_loop/evaluation/Tree_centers_eval/Compare/Eval'
# tree_error_path = '/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/project_andreas_zigzagentire/Correctness_loop/evaluation/Tree_centers_eval/Compare/Eval'
output_path_tree = Path(f'{tree_error_path}')
output_path_tree.mkdir(parents=True, exist_ok=True)

method_names = ['batch_aligned', 'icp', 'FGR', 'fastICP', 'robustICP','teaser++']
# method_names = ['icp', 'FGR', 'fastICP', 'robustICP','teaser++']

thresholds = [0.1, 0.5, 1, 10, 20, 50 ]
labels = ['Ours','ICP','FGR','FastICP','RobustICP','Teaser++']
# labels = ['ICP','FGR','FastICP','RobustICP','Teaser++']
percentage_below_threshold = {threshold: [] for threshold in thresholds}

err_tree_tosave = {}
for mn in method_names:
    err_tree_all_x_y = []
    err_tree_all_z = []
    err_tree_all_xyz = []

    for idx in range(1,3):
    # for idx in range(0):
        tree_error_file = f'{tree_error_path}/tree_error_{mn}_m{idx}.txt'
        err_data = np.loadtxt(tree_error_file, delimiter=',', skiprows=1)  # Skip the header
        err_tree_all_x_y.extend(err_data[:, 0])  # Column for x-y errors
        err_tree_all_z.extend(err_data[:, 1])  # Column for z errors
        err_tree_all_xyz.extend(err_data[:, 2])  # Column for xyz errors
    
    # Calculate the median of errors for each type
    median_x_y = np.median(err_tree_all_x_y)
    median_z = np.median(err_tree_all_z)
    median_xyz = np.median(err_tree_all_xyz)

    err_tree_tosave[mn] = [median_x_y, median_z, median_xyz]

    # Calculate the percentage of trees below each threshold for the current method
    for threshold in thresholds:
        percentage = (np.sum(np.array(err_tree_all_xyz) < threshold) / len(err_tree_all_xyz)) * 100
        percentage_below_threshold[threshold].append(percentage)

#save 
header = 'method,median_x-y,median_z,median_xyz'
data_to_save = np.array(list(err_tree_tosave.items()), dtype=object)
formatted_data = np.array([[method, *values] for method, values in data_to_save])

np.savetxt(f'{output_path_tree}/integrate_tree_error_m12.txt', formatted_data, fmt='%s', delimiter=',', header=header, comments='')


# # --- threshold
# # Plotting
# fig, ax = plt.subplots(figsize=(10, 7))
# for i, threshold in enumerate(thresholds):
#     ax.barh(np.arange(len(method_names)) + i * 0.1, percentage_below_threshold[threshold], height=0.1, label=f'< {threshold}')


# # Adjusting the plot
# ax.set_yticks(np.arange(len(method_names)) + 0.25)
# ax.set_yticklabels(labels)
# ax.set_xlabel('Percentage of Trees Below Threshold')
# ax.set_title('Percentage of Trees with XYZ Error Below Thresholds')
# ax.legend(title='Thresholds', loc='upper right')

# # plt.savefig(f'{output_path_tree}/threshold_m1.png')
# # plt.show()

fig, ax = plt.subplots(figsize=(14, 6)) 
# Calculate the width of each bar group
bar_width = 1.0 / (len(thresholds) + 1)  # Plus one to add some spacing between groups
group_width = bar_width * len(thresholds)
for i, threshold in enumerate(thresholds):
    # Calculate the starting X position of each group
    start_pos = np.arange(len(method_names)) * (group_width + bar_width) + i * bar_width
    # Plot vertical bars
    ax.bar(start_pos, percentage_below_threshold[threshold], width=bar_width, label=f'< {threshold}')
# Adjusting the plot
ax.set_xticks(np.arange(len(method_names)) * (group_width + bar_width) + (group_width / 2))
ax.set_xticklabels(labels, rotation=15, fontsize=24)  # Rotate labels to fit if necessary
ax.tick_params(axis='y', labelsize=22) 
ax.set_ylabel('Percentage (%)', fontsize=24)
ax.set_title('Percentage of Trees with XYZ Error Below Thresholds', fontsize=28)
ax.legend(title='Thresholds', loc='upper right', fontsize=20, title_fontsize=20)
plt.savefig(f'{output_path_tree}/threshold_m12.png')
plt.savefig(f'{output_path_tree}/threshold_m12.svg', format='svg')
plt.show()