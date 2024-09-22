""" This file is to show how to run the pipeline """

from subprocess import call
from pathlib import Path


# ----------------------------------------------------------------------------
#                                    Paths
# ----------------------------------------------------------------------------
PROJECT_PATH = f" "  # INPUT HERE THE PATH THE TO YOUR PROJECT FOLDER (create a folder where you want the output of this pipeline to be stored)

# Model_0
IMAGE_PATH_0 = f" "  # INPUT HERE THE PATH TO THE FOLDER CONTAINING THE IMAGES FOR MODEL 0
IM_LIST_0 = f" .txt"  # OPTIONAL: add a txt file containing a list of images that should be used from IMAGE_PATH_0
gt_cam_file_0 = f" .txt"  # INPUT HERE THE PATH TO THE GT FROM PIX4D
BIRD_EYE_PATH_0 = f" "  # INPUT HERE THE PATH TO THE BIRD EYE VIEWS (BEV) FOR MODEL 0 (GT)

# Model_1
IMAGE_PATH_1 = f" "  # INPUT HERE THE PATH TO THE FOLDER CONTAINING THE IMAGES FOR MODEL 1
IM_LIST_1 = f" .txt"  # OPTIONAL: add a txt file containing a list of images that should be used from IMAGE_PATH_0
gt_cam_file_1 = f" .txt"
BIRD_EYE_PATH_1 = f" "  # INPUT HERE THE PATH TO THE BIRD EYE VIEWS (BEV) FOR MODEL 1 (Predicted)

# Add additional query models as necessary.

# ----------------------------------------------------------------------------
#                                Pipeline
# ----------------------------------------------------------------------------

# Can add as many models as you want for step 1, but please first make sure there's no 'Images_LV95_batches.csv' before you start to collect data from model0
# 1. collect and transform image data
#  MODEL_0
create_image_batch_model0_command = [
    "python3",
    "Apple-Farm/collect_transform_image_data.py",
    f"{PROJECT_PATH}",
    f"{IMAGE_PATH_0}",
    "False",
    "-presorted_im", f"{IM_LIST_0}",
]
call(create_image_batch_model0_command)

# MODEL_1
create_image_batch_model1_command = [
    "python3",
    "Apple-Farm/collect_transform_image_data.py",
    f"{PROJECT_PATH}",
    f"{IMAGE_PATH_1}",
    "False",
    "-presorted_im", f"{IM_LIST_1}",
]
call(create_image_batch_model1_command)

# MODEL...


# 2. Reconstruction - Colmap sparse
colmap_cli_pipeline_command = [
    "python3",
    "Apple-Farm/colmap_cli_pipeline.py",
    f"{PROJECT_PATH}",
    f"{IMAGE_PATH_0}", f"{IMAGE_PATH_1}",  # If multiple image paths add in the order: image path model 0, image path model 1, ...
]
call(colmap_cli_pipeline_command)


# 3. Transform to LV95 and shift by const
transform_shifted_LV95_command = [
    "python3", 
    "Apple-Farm/transform_shifted_LV95.py", 
    f"{PROJECT_PATH}"
    ]
call(transform_shifted_LV95_command)


# 4. Feature Extraction Stage
# 4.1 Tree positions from BEV
tree_positions_command = [
    "python3",
    "Apple-Farm/tree_positions.py",
    f"{PROJECT_PATH}",
    f"{BIRD_EYE_PATH_0}", f"{BIRD_EYE_PATH_1}",  # If multiple image paths add in the order: BEV model 0, BEV model 1, ...
]
call(tree_positions_command)

# 4.2 Tree Segmentation
Tree_segment_command = [
    "python3", 
    "Apple-Farm/tree_area_segmentation.py", 
    f"{PROJECT_PATH}"]
call(Tree_segment_command)

# TODO: After finding the trunk, continue to process the slices below the trunk to separate the base of the tree from the ground
# 4.3 ground extraction for each tree segment
Ground_Extraction_command = [
    "python3", 
    "Apple-Farm/Gr_Ex_heightslice.py", 
    f"{PROJECT_PATH}"]
call(Ground_Extraction_command)


# 5. prepare for evaluation: shifted -> GT; trans base model to GT, others follow;
transform_init2gt_command = [
    "python3",
    "Apple-Farm/evaluation/estimation_LV95_gt_trans.py",
    f"{PROJECT_PATH}",
    "--image_lists", f"{IM_LIST_0}", f"{IM_LIST_1}",
    "--gt_cams", f"{gt_cam_file_0}", f"{gt_cam_file_1}",
    "--offset", " ", " ", " ",  # set offset (x,y,z) according to your dataset
]
call(transform_init2gt_command)


# 6. Initial alignment & feature extraction again
# 6.1 initial ICP
ICP_base_command = [
    "python3", 
    "Apple-Farm/ICP_base_model.py", 
    f"{PROJECT_PATH}"]
call(ICP_base_command)

# 6.2.1 Tree Segmentation
Tree_segment_command = [
    "python3", 
    "Apple-Farm/tree_seg_after_IICP.py", 
    f"{PROJECT_PATH}"]
call(Tree_segment_command)

# TODO: tree centers refinement;use mean of init trunk - to improve the trunk extracted
# 6.2.2 ground extraction for each tree segment
Ground_Extraction_command = [
    "python3", 
    "Apple-Farm/Gr_Ex_after_IICP.py", 
    f"{PROJECT_PATH}"]
call(Ground_Extraction_command)


# 7.1 Vertical correctness
ICP_vertical_command = [
    "python3", 
    "Apple-Farm/Loop_vertical_ICP.py", 
    f"{PROJECT_PATH}"]
# 7.2 Horizontal correctness
Horizontal_correctness_command = [
    "python3", 
    "Apple-Farm/Loop_horizontal_correctness.py", 
    f"{PROJECT_PATH}"]
# 7
SAVE_INTERVALS = "1,2,10"  # save the middle results
for i in range(1, 6):
    call(ICP_vertical_command + [str(i), SAVE_INTERVALS])
    scaling = "True"
    call(Horizontal_correctness_command + [str(i), SAVE_INTERVALS, scaling])


# 8. batch alignment
# 8.1 batch preparation [shifted to Aligned_models & batches separation]
batch_alignment_command = [
    "python3", 
    "Apple-Farm/batch_preparation.py", 
    f"{PROJECT_PATH}"]
call(batch_alignment_command)

# 8.2 batch_Horizontal correctness
Batch_Horizontal_correctness_command = [
    "python3",
    "Apple-Farm/batch_alignment.py",
    f"{PROJECT_PATH}",
    "-ver_ignore", "True",  # True: ignore the corresponding alignment;
    "-hor_ignore", "False",  # False: apply alignment
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
    "--data_path", f"{CAM_COMPARE_PATH}",
    "--output_path", f"{CAM_ERROR_PATH}",
]
call(cam_evaluation_command)


# 9.2 tree centers evaluation
# ablation study: steps along our pipeline
TREE_CENTERS_EVAL_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/Ours/Eval")
TREE_CENTERS_EVAL_PATH.mkdir(parents=True, exist_ok=True)

nr_batch = 2  # input here correct number of models (total number of base model and query models)
for idx in range(1, nr_batch):
    BASE_TREE = f"{PROJECT_PATH}/Correctness_loop/initial_alignment/Output/Fur_Ground_extraction/Model_0/trunk_centers.txt"
    initial_aligned = f"{PROJECT_PATH}/Correctness_loop/initial_alignment/Output/Fur_Ground_extraction/Model_{idx}/trunk_centers.txt"
    loop_aligned = f"{PROJECT_PATH}/Correctness_loop/2_horizontal/Output/Tree_centers/transformed_tree_pos_model_{idx}.txt"
    batch_aligned = f"{PROJECT_PATH}/Correctness_loop/3_batch_align/Horizontal_correct/tree_centers_m{idx}.csv"
    co_init = f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/Ours/colmap_init_tree{idx}.txt"

    tree_pairs_path = f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/tree_pairs_m{idx}.txt"

    EVAL_TREE_FILES = [co_init, initial_aligned, loop_aligned, batch_aligned]  #

    tree_evaluation_command = [
        "python3",
        "Apple-Farm/evaluation/trees_eval.py",
        "--ground_truth_positions", f"{BASE_TREE}",
        "--tree_positions", *EVAL_TREE_FILES,
        "--tree_pair_file", f"{tree_pairs_path}",
        "--output_path", f"{TREE_CENTERS_EVAL_PATH}",
        "--model_index", f"{idx}",
        "--method_names", "colmap_init", "initial aligned", "loop aligned", "batch aligned",
    ]
    call(tree_evaluation_command)

# SOTA comparison
TREE_CENTERS_EVAL_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/Compare/Eval")
TREE_CENTERS_EVAL_PATH.mkdir(parents=True, exist_ok=True)

nr_batch = 2
for idx in range(1, nr_batch):
    BASE_TREE = f"{PROJECT_PATH}/Correctness_loop/initial_alignment/Output/Fur_Ground_extraction/Model_0/trunk_centers.txt"
    folder = f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/Compare"
    icp = f"{folder}/ICP_m{idx}.txt"
    FGR = f"{folder}/FGR_m{idx}.txt"
    fastICP = f"{folder}/FICP_m{idx}.txt"
    robustICP = f"{folder}/RICP_m{idx}.txt"
    teaser = f"{folder}/teaser_m{idx}.txt"

    tree_pairs_path = f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/tree_pairs_m{idx}.txt"
    EVAL_TREE_FILES = [icp, FGR, fastICP, robustICP, teaser]
    tree_evaluation_command = [
        "python3",
        "Apple-Farm/evaluation/trees_eval.py",
        "--ground_truth_positions", f"{BASE_TREE}",
        "--tree_positions", *EVAL_TREE_FILES,
        "--tree_pair_file", f"{tree_pairs_path}",
        "--output_path", f"{TREE_CENTERS_EVAL_PATH}",
        "--model_index", f"{idx}",
        "--method_names", "icp", "FGR", "fastICP", "robustICP", "teaser++",
    ]
    call(tree_evaluation_command)