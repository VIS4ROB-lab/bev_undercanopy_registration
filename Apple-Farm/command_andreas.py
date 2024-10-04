from subprocess import call
from pathlib import Path
import os
import json

# ----------------------------------------------------------------------------
#                                    Paths
# ----------------------------------------------------------------------------
## zigzag
# Paths for local:
# INPUT HERE THE PATH THE TO YOUR PROJECT FOLDER (create a folder where you want the output of this pipeline to be stored)
PROJECT_PATH = "Dataset/Case_Opposite_View_andreas/Project_andreas"

# Model_0: 0130
IMAGE_PATH_0 = "Dataset/Case_Opposite_View_andreas/Data_andreas/andreas-park-20240130-images"  # INPUT HERE THE PATH TO THE FOLDER CONTAINING THE IMAGES FOR MODEL 1
IM_LIST_0 = "Dataset/Case_Opposite_View_andreas/Data_andreas/image_list_30zigzag.txt"  # OPTIONAL: add a txt file containing a list of images that should be used from IMAGE_PATH_0
gt_cam_file_0 = f"Dataset/Case_Opposite_View_andreas/Data_andreas/andreas-park-20240130-params/params/andreas-park-20240130-all_calibrated_external_camera_parameters.txt"
BIRD_EYE_PATH_0 = "Dataset/Case_Opposite_View_andreas/Data_andreas/bev_ground_truth_andreas_3001/ground_truth"  # INPUT HERE THE PATH TO THE BIRD EYE VIEWS (BEV) FOR MODEL
# BIRD_EYE_PATH_0 = "dataset_andreas_park/dynamic_kastelhof_traning_eval_andreas3001/dynamic_kastelhof_traning" # kastelhof training

# Model_1,2: 0127
IMAGE_PATH_1 = "Dataset/Case_Opposite_View_andreas/Data_andreas/andreas-park-20240127"  # INPUT HERE THE PATH TO THE FOLDER CONTAINING THE IMAGES FOR MODEL 0
IM_LIST_1 = "Dataset/Case_Opposite_View_andreas/Data_andreas/image_list_27zigzag_1.txt"  # OPTIONAL: add a txt file containing a list of images that should be used from IMAGE_PATH_0
gt_cam_file_1 = f"Dataset/Case_Opposite_View_andreas/Data_andreas/andreas-park-20240127-params/andreas-park-20240127-all_calibrated_external_camera_parameters.txt"
# BIRD_EYE_PATH_1 = "dataset_andreas_park/bev_ground_truth_andreas_2701/ground_truth" # GT BEV
BIRD_EYE_PATH_1 = "Dataset/Case_Opposite_View_andreas/Data_andreas/dynamic_andreas3001_training_eval_andreas2701/dynamic_3001_training"  # Andreas training
# BIRD_EYE_PATH_1 = "dataset_andreas_park/dynamic_kastelhof_traning_eval_andreas2701/dynamic_kastelhof_traning" # kastelhof training

IMAGE_PATH_2 = "Dataset/Case_Opposite_View_andreas/Data_andreas/andreas-park-20240127"  # INPUT HERE THE PATH TO THE FOLDER CONTAINING THE IMAGES FOR MODEL 0
IM_LIST_2 = "Dataset/Case_Opposite_View_andreas/Data_andreas/image_list_27zigzag_2.txt"  # OPTIONAL: add a txt file containing a list of images that should be used from IMAGE_PATH_0
gt_cam_file_2 = gt_cam_file_1
BIRD_EYE_PATH_2 = BIRD_EYE_PATH_1


# # #zigzag entire
# PROJECT_PATH = 'project_andreas_zigzagentire'

# # Model_0
# IMAGE_PATH_0 = 'dataset_andreas_park/andreas-park-20240127' # INPUT HERE THE PATH TO THE FOLDER CONTAINING THE IMAGES FOR MODEL 0
# IM_LIST_0 = 'dataset_andreas_park/image_list_27zigzag.txt' # OPTIONAL: add a txt file containing a list of images that should be used from IMAGE_PATH_0
# BIRD_EYE_PATH_0 = "dataset_andreas_park/bev_ground_truth_andreas_2701/ground_truth" # INPUT HERE THE PATH TO THE BIRD EYE VIEWS (BEV) FOR MODEL 0

# # Model_1
# IMAGE_PATH_1 = 'dataset_andreas_park/andreas-park-20240130-images' # INPUT HERE THE PATH TO THE FOLDER CONTAINING THE IMAGES FOR MODEL 1
# IM_LIST_1 = 'dataset_andreas_park/image_list_30zigzag.txt' # OPTIONAL: add a txt file containing a list of images that should be used from IMAGE_PATH_0
# BIRD_EYE_PATH_1 = "dataset_andreas_park/bev_ground_truth_andreas_3001/ground_truth" # INPUT HERE THE PATH TO THE BIRD EYE VIEWS (BEV) FOR MODEL 1


# ----------------------------------------------------------------------------
#                                Pipeline
# ----------------------------------------------------------------------------
# --- 0. Loading config file
config_file = 'Apple-Farm/config/config_andreas.json'
with open(config_file, 'r') as file:
    config = json.load(file)
nr_batch = config["nr_models"]

# --- 1. collect and transform image data
# Can add as many models as you want for step 1

# Check if the file 'Images_LV95_batches.csv' exists and delete if it does
file_path = os.path.join(PROJECT_PATH, 'Images_LV95_batches.csv')
if os.path.exists(file_path):
    os.remove(file_path)

# MODEL_0
create_image_batch_model0_command = [
    "python3",
    "Apple-Farm/src/collect_transform_image_data.py",
    f"{PROJECT_PATH}",
    f"{IMAGE_PATH_0}",
    config['create_image_batch_model0_command']['cluster_images'],
] +(["-presorted_im", f"{IM_LIST_0}"] if IM_LIST_0 else []) \
  +(["-offset"] + list(map(str, config['offset'])) if 'offset' in config and config['offset'] else []) \
  + (["-y_boundary", f"{config['create_image_batch_model0_command']['y_boundary']}"] if 'y_boundary' in config['create_image_batch_model0_command'] else []) \
  + (["-x_boundary", f"{config['create_image_batch_model0_command']['x_boundary']}"] if 'x_boundary' in config['create_image_batch_model0_command'] else [])
call(create_image_batch_model0_command)

# MODEL_1
create_image_batch_model1_command = [
    "python3",
    "Apple-Farm/src/collect_transform_image_data.py",
    f"{PROJECT_PATH}",
    f"{IMAGE_PATH_1}",
    config['create_image_batch_model1_command']['cluster_images'],
] +(["-presorted_im", f"{IM_LIST_1}"] if IM_LIST_1 else []) \
  +(["-offset"] + list(map(str, config['offset'])) if 'offset' in config and config['offset'] else []) \
  + (["-y_boundary", f"{config['create_image_batch_model0_command']['y_boundary']}"] if 'y_boundary' in config['create_image_batch_model0_command'] else []) \
  + (["-x_boundary", f"{config['create_image_batch_model0_command']['x_boundary']}"] if 'x_boundary' in config['create_image_batch_model0_command'] else [])
call(create_image_batch_model1_command)

# MODEL_2
create_image_batch_model2_command = [
    "python3",
    "Apple-Farm/src/collect_transform_image_data.py",
    f"{PROJECT_PATH}",
    f"{IMAGE_PATH_2}",
    config['create_image_batch_model2_command']['cluster_images'],
] +(["-presorted_im", f"{IM_LIST_2}"] if IM_LIST_2 else []) \
  +(["-offset"] + list(map(str, config['offset'])) if 'offset' in config and config['offset'] else []) \
  + (["-y_boundary", f"{config['create_image_batch_model0_command']['y_boundary']}"] if 'y_boundary' in config['create_image_batch_model0_command'] else []) \
  + (["-x_boundary", f"{config['create_image_batch_model0_command']['x_boundary']}"] if 'x_boundary' in config['create_image_batch_model0_command'] else [])
call(create_image_batch_model2_command)


# # --- (Skip this step given the point cloud is already provided.) 2. Colmap CLI pipeline sparse
# colmap_cli_pipeline_command = [
#     "python3",
#     "Apple-Farm/src/colmap_cli_pipeline.py",
#     f"{PROJECT_PATH}",
#     f"{IMAGE_PATH_0}",
#     f"{IMAGE_PATH_1}",
#     f"{IMAGE_PATH_2}",  # If multiple image paths add in the order: image path model 0, image path model 1, ...
# ]
# call(colmap_cli_pipeline_command)


# --- 3. Transform to LV95 and shift by const
transform_shifted_LV95_command = ["python3", "Apple-Farm/src/transform_shifted_LV95.py", f"{PROJECT_PATH}"]
call(transform_shifted_LV95_command)


# --- 4. Feature Stage
# 4.1 Tree positions
tree_positions_comand = [
    "python3",
    "Apple-Farm/src/tree_positions.py",
    f"{PROJECT_PATH}",
    f"{BIRD_EYE_PATH_0}",
    f"{BIRD_EYE_PATH_1}",
    f"{BIRD_EYE_PATH_2}", 
]+(["-cluster_rad", f"{config['tree_positions_comand']['cluster_rad']}"] if 'cluster_rad' in config['tree_positions_comand'] else []) 
call(tree_positions_comand)

# 4.2 Tree Segmentation
Tree_segment_command = [
    "python3",
    "Apple-Farm/src/tree_area_segmentation.py",
    f"{PROJECT_PATH}",
] + (["-tree_area_dist", f'{config["tree_segment_command"]["tree_area_dist"]}'] if 'tree_area_dist' in config['tree_segment_command'] else []) \
  + (["-removing_tree_radius", f'{config["tree_segment_command"]["removing_tree_radius"]}'] if 'removing_tree_radius' in config['tree_segment_command'] else [])
call(Tree_segment_command)

# 4.3 ground extraction for each tree segment
Ground_Extraction_command = [
    "python3",
    "Apple-Farm/src/Gr_Ex_heightslice.py",
    f"{PROJECT_PATH}",
] + (["-slice_interval", f'{config["Ground_Extraction_command"]["slice_interval"]}'] if 'slice_interval' in config['Ground_Extraction_command'] else []) \
  + (["-expected_mean_tolerance", f'{config["Ground_Extraction_command"]["expected_mean_tolerance"]}'] if 'expected_mean_tolerance' in config['Ground_Extraction_command'] else []) \
  + (["-center_area", f'{config["Ground_Extraction_command"]["center_area"]}'] if 'center_area' in config['Ground_Extraction_command'] else [])
call(Ground_Extraction_command)


# --- 5. evaluation preparation: shifted -> GT; trans base model to GT, others follow;
transform_init2gt_command = [
    "python3",
    "Apple-Farm/evaluation/estimation_LV95_gt_trans.py",
    f"{PROJECT_PATH}",
    "--image_lists",
    f"{IM_LIST_0}", f"{IM_LIST_1}", f"{IM_LIST_2}",
    "--gt_cams",
    f"{gt_cam_file_0}", f"{gt_cam_file_1}", f"{gt_cam_file_2}",
] +(["--offset"] + list(map(str, config['offset'])) if config['offset'] else [])
call(transform_init2gt_command)


# --- 6. Initial Alignment & Further Feature Extraction
# 6.1 initial ICP
ICP_base_command = ["python3", "Apple-Farm/src/ICP_base_model.py", f"{PROJECT_PATH}"]
call(ICP_base_command)

# 6.2.1 Tree Segmentation
Further_Tree_segment_command = ["python3", "Apple-Farm/src/tree_seg_after_IICP.py", f"{PROJECT_PATH}",]\
                        + (["-tree_area_dist", f'{config["Further_Tree_segment_command"]["tree_area_dist"]}'] if 'tree_area_dist' in config['Further_Tree_segment_command'] else [])\
                        + (["-removing_tree_radius", f'{config["Further_Tree_segment_command"]["removing_tree_radius"]}'] if 'removing_tree_radius' in config['Further_Tree_segment_command'] else [])\
                        + (["-boundary", f'{config["Further_Tree_segment_command"]["boundary"]}'] if 'boundary' in config['Further_Tree_segment_command'] else [])
call(Further_Tree_segment_command)

# 6.2.2 ground extraction for each tree segment
Further_Ground_Extraction_command = [
    "python3",
    "Apple-Farm/src/Gr_Ex_after_IICP.py",
    f"{PROJECT_PATH}",
] + (["-slice_interval", f'{config["Further_Ground_Extraction_command"]["slice_interval"]}'] if 'slice_interval' in config['Further_Ground_Extraction_command'] else []) \
    +(["-expected_mean_tolerance", f'{config["Further_Ground_Extraction_command"]["expected_mean_tolerance"]}'] if 'expected_mean_tolerance' in config['Further_Ground_Extraction_command'] else []) \
    +(["-center_area", f'{config["Further_Ground_Extraction_command"]["center_area"]}'] if 'center_area' in config['Further_Ground_Extraction_command'] else [])\
    +(["-expected_trunk_radius", f'{config["Further_Ground_Extraction_command"]["expected_trunk_radius"]}'] if 'expected_trunk_radius' in config['Further_Ground_Extraction_command'] else [])
call(Further_Ground_Extraction_command)


# --- 7. Loop Alignment
# 7.1 Vertical correctness
ICP_vertical_command = ["python3", "Apple-Farm/src/Loop_vertical_ICP.py", f"{PROJECT_PATH}",]\
                        +(["--icp_threshold", f'{config["ICP_vertical_command"]["icp_threshold"]}'] if 'icp_threshold' in config['ICP_vertical_command'] else [])
# 7.2 Horizontal correctness
Horizontal_correctness_command = [
    "python3",
    "Apple-Farm/src/Loop_horizontal_correctness.py",
    f"{PROJECT_PATH}",
] + (["-max_tree_pair_dist", f'{config["Horizontal_correctness_command"]["max_tree_pair_dist"]}'] if 'max_tree_pair_dist' in config["Horizontal_correctness_command"] else [])
# 7
SAVE_INTERVALS = "1,2,10"  # "1,2,5,10" save the middle results
for i in range(1, 6):
    call(ICP_vertical_command + [str(i), SAVE_INTERVALS])
    scaling = "True"
    call(Horizontal_correctness_command + [str(i), SAVE_INTERVALS, scaling])


# --- 8. Batch Alignment
# 8.1 batch alignment [shifted to Aligned_models & batches separation]
batch_alignment_command = ["python3", "Apple-Farm/src/batch_preparation.py", f"{PROJECT_PATH}"]\
                            + (["-similarity_threshold", f'{config["batch_alignment_command"]["similarity_threshold"]}'] if 'similarity_threshold' in config['batch_alignment_command'] else [])\
                            + (["-tree_pair_distance", f'{config["batch_alignment_command"]["tree_pair_distance"]}'] if 'tree_pair_distance' in config['batch_alignment_command'] else [])
call(batch_alignment_command)

# 8.2 batch_Horizontal correctness (with model)
Batch_Horizontal_correctness_command = [
    "python3",
    "Apple-Farm/src/batch_alignment.py",
    f"{PROJECT_PATH}",
] + (["-ver_ignore", "True"] if 'ver_ignore' in config['Batch_Horizontal_correctness_command'] and config['Batch_Horizontal_correctness_command']['ver_ignore'] else [])\
  + (["-hor_ignore", "True"] if 'hor_ignore' in config['Batch_Horizontal_correctness_command'] and config['Batch_Horizontal_correctness_command']['hor_ignore'] else [])\
  + (["-max_tree_pair_dist", f'{config["Batch_Horizontal_correctness_command"]["max_tree_pair_dist"]}'] if 'max_tree_pair_dist' in config['Batch_Horizontal_correctness_command'] else [])
call(Batch_Horizontal_correctness_command)

# 8.3 camera assignment &
Camera_assignment_command = ["python3", "Apple-Farm/src/batch_cam_assign.py", f"{PROJECT_PATH}"]
call(Camera_assignment_command)


# --- 9. Evaluation
# 9.1 cam_poses_error T & R evaluation
CAM_COMPARE_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/Camera_poses_compare")
CAM_ERROR_PATH = Path(f"{CAM_COMPARE_PATH}/Cam_pose_Error")
CAM_ERROR_PATH.mkdir(parents=True, exist_ok=True)
cam_evaluation_command = [
    "python3",
    "Apple-Farm/evaluation/cam_error_evaluation.py",
    "--data_path",
    f"{CAM_COMPARE_PATH}",
    "--output_path",
    f"{CAM_ERROR_PATH}",
    "--nr_models", f"{nr_batch}"
]
call(cam_evaluation_command)

# 9.2 tree centers evaluation (with pair index)
# ablation study: steps along our pipeline
TREE_CENTERS_EVAL_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/Ours/Eval")
TREE_CENTERS_EVAL_PATH.mkdir(parents=True, exist_ok=True)

for idx in range(1, nr_batch):
    BASE_TREE = (
        f"{PROJECT_PATH}/Correctness_loop/initial_alignment/Output/Fur_Ground_extraction/Model_0/trunk_centers.txt"
    )
    initial_aligned = (
        f"{PROJECT_PATH}/Correctness_loop/initial_alignment/Output/Fur_Ground_extraction/Model_{idx}/trunk_centers.txt"
    )
    loop_aligned = (
        f"{PROJECT_PATH}/Correctness_loop/2_horizontal/Output/Tree_centers/transformed_tree_pos_model_{idx}.txt"
    )
    batch_aligned = f"{PROJECT_PATH}/Correctness_loop/3_batch_align/Horizontal_correct/tree_centers_m{idx}.csv"
    co_init = f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/Ours/colmap_init_tree{idx}.txt"

    tree_pairs_path = f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/tree_pairs_m{idx}.txt"

    EVAL_TREE_FILES = [co_init, initial_aligned, loop_aligned, batch_aligned]

    tree_evaluation_command = [
        "python3",
        "Apple-Farm/evaluation/trees_eval.py",
        "--ground_truth_positions",
        f"{BASE_TREE}",
        "--tree_positions",
        *EVAL_TREE_FILES,
        "--tree_pair_file",
        f"{tree_pairs_path}",
        "--output_path",
        f"{TREE_CENTERS_EVAL_PATH}",
        "--model_index",
        f"{idx}",
        "--method_names",
        "colmap_init",
        "initial aligned",
        "loop aligned",
        "batch aligned",
    ]
    call(tree_evaluation_command)

# # If needed: SOTA comparison
# TREE_CENTERS_EVAL_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/Compare/Eval")
# TREE_CENTERS_EVAL_PATH.mkdir(parents=True, exist_ok=True)

# for idx in range(1, nr_batch):
#     BASE_TREE = (
#         f"{PROJECT_PATH}/Correctness_loop/initial_alignment/Output/Fur_Ground_extraction/Model_0/trunk_centers.txt"
#     )
#     folder = f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/Compare"
#     icp = f"{folder}/ICP_m{idx}.txt"
#     FGR = f"{folder}/FGR_m{idx}.txt"
#     fastICP = f"{folder}/FICP_m{idx}.txt"
#     robustICP = f"{folder}/RICP_m{idx}.txt"
#     teaser = f"{folder}/teaser_m{idx}.txt"

#     tree_pairs_path = f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/tree_pairs_m{idx}.txt"
#     EVAL_TREE_FILES = [icp, FGR, fastICP, robustICP, teaser]
#     tree_evaluation_command = [
#         "python3",
#         "Apple-Farm/evaluation/trees_eval.py",
#         "--ground_truth_positions",
#         f"{BASE_TREE}",
#         "--tree_positions",
#         *EVAL_TREE_FILES,
#         "--tree_pair_file",
#         f"{tree_pairs_path}",
#         "--output_path",
#         f"{TREE_CENTERS_EVAL_PATH}",
#         "--model_index",
#         f"{idx}",
#         "--method_names",
#         "icp",
#         "FGR",
#         "fastICP",
#         "robustICP",
#         "teaser++",
#     ]
#     call(tree_evaluation_command)
