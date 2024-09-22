import pandas as pd
import numpy as np
from pathlib import Path
import shutil

 # INPUT HERE THE PATH THE TO YOUR PROJECT FOLDER 
# PROJECT_PATH = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case2/project_seasons'
# PROJECT_PATH = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/project_new_andreas_zigzagpair'
# PROJECT_PATH = '/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case1/project_views'
PROJECT_PATH = "/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/project_andreas_zigzagentire"

EVALUATION_DIR = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation")

idx = 2
# -------------- comparison ----------------------
INPUT_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/Ours")
OUTPUT_PATH = Path(f"{PROJECT_PATH}/Correctness_loop/evaluation/Tree_centers_eval/Compare")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# co_init_file = f'{INPUT_PATH}/colmap_init_tree{idx}.txt'
co_init_file = f"{INPUT_PATH}/co_init_m0_b{idx}.txt"  # zigzag-entire
tree_df = pd.read_csv(co_init_file, delimiter=" ", header=0)
idx_tree = tree_df["tree_idx"].tolist()
x_tree = tree_df["x"].tolist()
y_tree = tree_df["y"].tolist()
z_tree = tree_df["z"].tolist()

# andreas zigzag pairs
# trans_file = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/project_new_andreas_zigzagpair/Correctness_loop/evaluation/comparison_method/model{idx}_bd/ICP_txt/transformation.txt'
# trans_file = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/project_new_andreas_zigzagpair/Correctness_loop/evaluation/comparison_method/model{idx}_bd/FGR_txt/transformation.txt'
# trans_file = f'/home/zjw/SP/forest_3d_sarah_lia/Fast-Robust-ICP/data/andreas_zigzag/model{idx}_m2trans.txt'
# trans_file = f'/home/zjw/SP/forest_3d_sarah_lia/Fast-Robust-ICP/data/andreas_zigzag/model{idx}_m3trans.txt'
# trans_file = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/project_new_andreas_zigzagpair/Correctness_loop/evaluation/teaser/trans_teaser_bounded_m{idx}.txt'

# diff_seasons
# trans_file = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case2/project_seasons/Correctness_loop/evaluation/comparison_method/ICP_txt/transformation.txt'
# trans_file = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case2/project_seasons/Correctness_loop/evaluation/comparison_method/FGR_txt/transformation.txt'
# trans_file = f'/home/zjw/SP/forest_3d_sarah_lia/Fast-Robust-ICP/data/diff_seasons/outputm2trans.txt'
# trans_file = f'/home/zjw/SP/forest_3d_sarah_lia/Fast-Robust-ICP/data/diff_seasons/outputm3trans.txt'
# trans_file = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case2/project_seasons/Correctness_loop/evaluation/comparison_method/teaser/trans_teaser_bounded.txt'

# # diff_views
# trans_file = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case1/project_views/Correctness_loop/evaluation/comparison_method/ICP_txt/transformation_m{idx}.txt'
# trans_file = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case1/project_views/Correctness_loop/evaluation/comparison_method/FGR_txt/transformation_m{idx}.txt'
# trans_file = f'/home/zjw/SP/forest_3d_sarah_lia/Fast-Robust-ICP/data/diff_views/output/model{idx}_m3trans.txt'
# trans_file = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case1/project_views/Correctness_loop/evaluation/comparison_method/teaser/trans_teaser_bounded_m{idx}.txt'


# zigzag entire
# trans_file = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/project_andreas_zigzagentire/Correctness_loop/evaluation/comparison_method/ICP_txt/transformation.txt'
# trans_file = f'/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/project_andreas_zigzagentire/Correctness_loop/evaluation/comparison_method/FGR_txt/transformation.txt'
# trans_file = f'/home/zjw/SP/forest_3d_sarah_lia/Fast-Robust-ICP/data/andreas_zigzag_entire/outputm3trans.txt'
trans_file = f"/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/project_andreas_zigzagentire/Correctness_loop/evaluation/teaser/trans_teaser_bounded_m0.txt"

ICP_tran = np.loadtxt(trans_file)

coord_trees = np.vstack((x_tree, y_tree, z_tree, np.ones(len(x_tree))))
gt_transed_coord_trees = ICP_tran @ coord_trees
x_tree_transformed, y_tree_transformed, z_tree_transformed = gt_transed_coord_trees[:3]

transformed_trees = np.vstack((idx_tree, x_tree_transformed, y_tree_transformed, z_tree_transformed)).T
transformed_tree_df = pd.DataFrame(transformed_trees, columns=["tree_idx", "x", "y", "z"])
transformed_tree_df["tree_idx"] = transformed_tree_df["tree_idx"].astype(int)
transformed_tree_df.to_csv(f"{OUTPUT_PATH}/teaser_m{idx}.txt", sep=" ", index=False, header=True)
# ICP FGR FICP RICP teaser