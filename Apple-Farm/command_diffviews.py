""" This file is to show how to run the pipeline """

from subprocess import call
from pathlib import Path


# ----------------------------------------------------------------------------
#                                    Paths
# ----------------------------------------------------------------------------
# Paths for local:
PROJECT_PATH = "case1/project_views"  # INPUT HERE THE PATH THE TO YOUR PROJECT FOLDER (create a folder where you want the output of this pipeline to be stored)

# Model_0: winter
IMAGE_PATH_0 = "case1/winter/images_winter_all"  # INPUT HERE THE PATH TO THE FOLDER CONTAINING THE IMAGES FOR MODEL 0
IM_LIST_0 = "case1/winter/image_list_winter.txt"  # OPTIONAL: add a txt file containing a list of images that should be used from IMAGE_PATH_0
gt_cam_file_0 = f"case1/winter/applefarm-230209_merge_v9_calibrated_external_camera_parameters.txt"
BIRD_EYE_PATH_0 = "case1/winter/BEV_Kastelhof_Winter_gt"  # INPUT HERE THE PATH TO THE BIRD EYE VIEWS (BEV) FOR MODEL 0
# BIRD_EYE_PATH_0 = f'case1/winter/BEV_Winter_oringinal'  # ORIGINAL

# Model_1: summer_east
IMAGE_PATH_1 = "case1/image_summer_all"  # INPUT HERE THE PATH TO THE FOLDER CONTAINING THE IMAGES FOR MODEL 1
IM_LIST_1 = "case1/summer_east/image_list_east.txt"  # OPTIONAL: add a txt file containing a list of images that should be used from IMAGE_PATH_0
gt_cam_file_1 = f"case1/summer_east/apple-farm-230209-east_calibrated_external_camera_parameters.txt"
# BIRD_EYE_PATH_1 = "case1/summer_east/BEV_Kastelhof_Summer_east_gt" # INPUT HERE THE PATH TO THE BIRD EYE VIEWS (BEV) FOR MODEL 1
BIRD_EYE_PATH_1 = "case2/summer/bev_dynamic_kastelhof_summer"

# Model_2: summer_north
IMAGE_PATH_2 = "case1/image_summer_all"  # INPUT HERE THE PATH TO THE FOLDER CONTAINING THE IMAGES FOR MODEL 1
IM_LIST_2 = "case1/summer_north/image_list_north.txt"  # OPTIONAL: add a txt file containing a list of images that should be used from IMAGE_PATH_0
gt_cam_file_2 = f"case1/summer_north/apple-farm-20230627-north_calibrated_external_camera_parameters.txt"
# BIRD_EYE_PATH_2 = "case1/summer_north/BEV_Kastelhof_Summer_north_gt" # INPUT HERE THE PATH TO THE BIRD EYE VIEWS (BEV) FOR MODEL 1
BIRD_EYE_PATH_2 = "case2/summer/bev_dynamic_kastelhof_summer"
# ------------

# Can add as many models as you want for step 1, but first delete 'Images_LV95_batches.csv' and regenerate it with code below if you want to start from model0 again
# 1. collect and transform image data
#  MODEL_0
create_image_batch_model0_command = [
    "python3",
    "Apple-Farm/collect_transform_image_data.py",
    f"{PROJECT_PATH}",
    f"{IMAGE_PATH_0}",
    "False",
    "-presorted_im",
    f"{IM_LIST_0}",  # Optional list of images to use
]
call(create_image_batch_model0_command)

# MODEL_1
create_image_batch_model1_command = [
    "python3",
    "Apple-Farm/collect_transform_image_data.py",
    f"{PROJECT_PATH}",
    f"{IMAGE_PATH_1}",
    "False",
    "-presorted_im",
    f"{IM_LIST_1}",
]
call(create_image_batch_model1_command)

# MODEL_2
create_image_batch_model2_command = [
    "python3",
    "Apple-Farm/collect_transform_image_data.py",
    f"{PROJECT_PATH}",
    f"{IMAGE_PATH_2}",
    "False",
    "-presorted_im",
    f"{IM_LIST_2}",
]
call(create_image_batch_model2_command)


# 2. Reconstruction - Colmap sparse
colmap_cli_pipeline_command = [
    "python3",
    "Apple-Farm/colmap_cli_pipeline.py",
    f"{PROJECT_PATH}",
    f"{IMAGE_PATH_0}",
    f"{IMAGE_PATH_1}",
    f"{IMAGE_PATH_2}",  # If multiple image paths add in the order: image path model 0, image path model 1, ...
]
call(colmap_cli_pipeline_command)


# 3. Transform to LV95 and shift by const
transform_shifted_LV95_command = ["python3", "Apple-Farm/transform_shifted_LV95.py", f"{PROJECT_PATH}"]
call(transform_shifted_LV95_command)


# 4. Feature Extraction Stage
# 4.1 Tree positions from BEV
tree_positions_command = [
    "python3",
    "Apple-Farm/tree_positions.py",
    f"{PROJECT_PATH}",
    f"{BIRD_EYE_PATH_0}",
    f"{BIRD_EYE_PATH_1}",
    f"{BIRD_EYE_PATH_2}",  # If multiple image paths add in the order: BEV model 0, BEV model 1, ...
]
call(tree_positions_command)

# 4.2 Tree Segmentation
Tree_segment_command = [
    "python3",
    "Apple-Farm/tree_area_segmentation.py",
    f"{PROJECT_PATH}",
    "-tree_area_dist",
    f"{8.0}",
]
call(Tree_segment_command)

# 4.3 ground extraction for each tree segment
Ground_Extraction_command = [
    "python3",
    "Apple-Farm/Gr_Ex_heightslice.py",
    f"{PROJECT_PATH}",
    "-center_area",
    f"{4}",
    "-expected_mean_tolerance",
    f"{2.5}",
]
call(Ground_Extraction_command)


# 5. evaluation: shifted -> GT; trans base model to GT, others follow;
transform_init2gt_command = [
    "python3",
    "Apple-Farm/evaluation/estimation_LV95_gt_trans.py",
    f"{PROJECT_PATH}",
    "--image_lists",
    f"{IM_LIST_0}",
    f"{IM_LIST_1}",
    f"{IM_LIST_2}",
    "--gt_cams",
    f"{gt_cam_file_0}",
    f"{gt_cam_file_1}",
    f"{gt_cam_file_2}",
    "--offset",
    "2678199.0",
    "1258409.0",
    "445.000",
]
call(transform_init2gt_command)


# 6. Initial alignment & feature extraction again
# 6.1 initial ICP
ICP_base_command = ["python3", "Apple-Farm/ICP_base_model.py", f"{PROJECT_PATH}"]
call(ICP_base_command)

# 6.2.1 Tree Segmentation
Tree_segment_command = ["python3", "Apple-Farm/tree_seg_after_IICP.py", f"{PROJECT_PATH}", "-tree_area_dist", f"{8.0}"]
call(Tree_segment_command)
# TODO: tree centers refinement
# 6.2.2 ground extraction for each tree segment
Ground_Extraction_command = [
    "python3",
    "Apple-Farm/Gr_Ex_after_IICP.py",
    f"{PROJECT_PATH}",
    "-slice_interval",
    f"{0.3}",  # 0.4
    "-expected_mean_tolerance",
    "0.6",
]
call(Ground_Extraction_command)


# 7.1 Vertical correctness
ICP_vertical_command = ["python3", "Apple-Farm/Loop_vertical_ICP.py", f"{PROJECT_PATH}", "--icp_threshold", "0.5"]
# 7.2 Horizontal correctness
Horizontal_correctness_command = [
    "python3",
    "Apple-Farm/Loop_horizontal_correctness.py",
    f"{PROJECT_PATH}",
    "-max_tree_pair_dist",
    "4",
]
# 7
SAVE_INTERVALS = "1,2,10"  # save the middle results
for i in range(1, 6):
    call(ICP_vertical_command + [str(i), SAVE_INTERVALS])
    scaling = "True"
    call(Horizontal_correctness_command + [str(i), SAVE_INTERVALS, scaling])


# 8.1 batch alignment [shifted to Aligned_models & batches separation]
batch_alignment_command = [
    "python3",
    "Apple-Farm/batch_preparation.py",
    f"{PROJECT_PATH}",
    "-similarity_threshold",
    "0.6",
]
call(batch_alignment_command)

# 8.2 batch_Horizontal correctness
Batch_Horizontal_correctness_command = [
    "python3",
    "Apple-Farm/batch_alignment.py",
    f"{PROJECT_PATH}",
    "-ver_ignore",
    "True",  # True: ignore the corresponding alignment;
    "-hor_ignore",
    "False",  # False: apply alignment
]
call(Batch_Horizontal_correctness_command)

# 8.3 camera assignment &
Camera_assignment_command = ["python3", "Apple-Farm/batch_cam_assign.py", f"{PROJECT_PATH}"]
call(Camera_assignment_command)


# 9.1 cam_poses_error T & R evaluation
CAM_COMPARE_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/Camera_poses_compare")
CAM_ERROR_PATH = Path(f"{CAM_COMPARE_PATH}/Cam_pose_Error")
CAM_ERROR_PATH.mkdir(parents=True, exist_ok=True)
cam_evaluation_command = [
    "python3",
    "Apple-Farm/cam_error_evaluation.py",
    "--data_path",
    f"{CAM_COMPARE_PATH}",
    "--output_path",
    f"{CAM_ERROR_PATH}",
]
call(cam_evaluation_command)


# 9.2 tree centers evaluation (w/ pair index)
TREE_CENTERS_EVAL_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/Ours/Eval")
TREE_CENTERS_EVAL_PATH.mkdir(parents=True, exist_ok=True)

nr_batch = 3
for idx in range(1, nr_batch):
    tree_pairs_path = f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/tree_pairs_m{idx}.txt"

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

    EVAL_TREE_FILES = [co_init, initial_aligned, loop_aligned, batch_aligned]  #

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


TREE_CENTERS_EVAL_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/Compare/Eval")
TREE_CENTERS_EVAL_PATH.mkdir(parents=True, exist_ok=True)
nr_batch = 3
for idx in range(1, nr_batch):
    tree_pairs_path = f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/tree_pairs_m{idx}.txt"

    BASE_TREE = (
        f"{PROJECT_PATH}/Correctness_loop/initial_alignment/Output/Fur_Ground_extraction/Model_0/trunk_centers.txt"
    )
    folder = f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/Compare"
    icp = f"{folder}/ICP_m{idx}.txt"
    FGR = f"{folder}/FGR_m{idx}.txt"
    fastICP = f"{folder}/FICP_m{idx}.txt"
    robustICP = f"{folder}/RICP_m{idx}.txt"
    teaser = f"{folder}/teaser_m{idx}.txt"

    EVAL_TREE_FILES = [icp, FGR, fastICP, robustICP, teaser]
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
        "icp",
        "FGR",
        "fastICP",
        "robustICP",
        "teaser++",
    ]
    call(tree_evaluation_command)
