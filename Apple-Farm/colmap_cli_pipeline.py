# Colmap CLI pipeline
import argparse
from subprocess import call, run
from pathlib import Path
import os
import shutil
import csv
import pandas as pd


# ------------------------------------------------------------------
# -                           ARGPARSE                             -
# ------------------------------------------------------------------

# Argparse: Create parser
parser = argparse.ArgumentParser()
# Argparse: Add arguments
parser.add_argument(dest="project_path", help="Path to the project folder")
parser.add_argument(
    dest="images_path",
    nargs="+",
    help="Path to the images. If multiple image paths, provide them in the order of the models they belong to (Path/to/Model_0 Path/to/Model_2 ...)",
)
parser.add_argument(
    "--colmap_command",
    dest="COLMAP",
    default="colmap",
    help="Change if colmap is installed locally at specific path ./path/to/colmap",
)
# Argparse: Parse arguments
args = parser.parse_args()
COLMAP = args.COLMAP


# ------------------------------------------------------------------
# -                           Set up                               -
# ------------------------------------------------------------------

IMAGES_LV95_BATCHES = f"{args.project_path}/Images_LV95_batches.csv"

# Read number of batches/models
df_batches = pd.read_csv(IMAGES_LV95_BATCHES)
max_batch = df_batches["batch"].max()
nr_batches = int(max_batch) + 1
print(nr_batches)

# Create Models path (output path)
output_path = Path(f"{args.project_path}/Models")
output_path.mkdir(parents=True, exist_ok=True)

# Check if multiple paths where passed
if len(args.images_path) == 1:
    IMAGES_PATH = Path(args.images_path[0])
    multiple_im_paths = False
else:
    multiple_im_paths = True


# Check if reconstruction log exists
recon_log_path = Path(f"{output_path}/recon_log.csv")
file_exists = os.path.exists(recon_log_path)
# print("does the file exist? ",file_exists)

# Create reconstruction log file if does not exist already
if not file_exists:

    print("Creating log file...")

    with open(recon_log_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Model", "status"])

        # write initial model status
        for idx in range(nr_batches):
            writer.writerow([f"Model_{idx}", "0"])

        print(f"Reconstruction log file created at '{recon_log_path}'")

log_df = pd.read_csv(recon_log_path)  # write recon log in df


# ------------------------------------------------------------------
# -                       Reconstruction                           -
# ------------------------------------------------------------------

# Loop through image batches to create models
for idx in range(nr_batches):

    # Check if model is already reconstructed
    current_model = f"Model_{idx}"
    model_row = log_df[log_df["Model"] == current_model]
    status = bool(model_row["status"].iloc[0])

    # If done ignore
    if status == True:
        print(current_model, "is already done")

    # If not done: do reconstruction (SfM) using Colmap
    else:
        print(current_model, "NOT DONE...........................................................")

        if multiple_im_paths:
            IMAGES_PATH = Path(args.images_path[idx])

        # Create output directory
        OUTPUT_PATH = Path(f"{output_path}/Model_{idx}")
        # Delete if there was a folder created with this name
        if os.path.exists(OUTPUT_PATH):
            shutil.rmtree(OUTPUT_PATH)
            print(f"Directory '{OUTPUT_PATH}' deleted.")
        OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

        # Output directory
        DATASET_PATH = f"{output_path}/Model_{idx}"

        # Create images_list_paths.txt to pass to Colmap pipeline
        names_images = df_batches.loc[df_batches["batch"] == idx, "name"].tolist()
        IMAGE_LIST = f"{DATASET_PATH}/image_list_path.txt"
        with open(IMAGE_LIST, "w") as file:
            for image_name in names_images:
                file.write(image_name + "\n")

        # COLMAP CLI Reconstruction pipeline
        # ---------------------------------------------

        # Extract features:
        feature_extractor_command = [
            f"{COLMAP}",
            "feature_extractor",
            "--image_path",
            f"{IMAGES_PATH}",
            "--image_list_path",
            f"{IMAGE_LIST}",
            "--database_path",
            f"{DATASET_PATH}/database.db",
            "--SiftExtraction.max_image_size",
            "2400",
        ]
        call(feature_extractor_command)

        # Matching
        exhaustive_matcher_command = [
            f"{COLMAP}",
            "exhaustive_matcher",  # sequential match
            "--database_path",
            f"{DATASET_PATH}/database.db",
            "--SiftMatching.guided_matching",
            "1",
        ]
        call(exhaustive_matcher_command)

        SPARSE_PATH = f"{DATASET_PATH}/sparse"
        run(["mkdir", SPARSE_PATH], check=True)

        # Mapping
        mapper_command = [
            f"{COLMAP}",
            "mapper",
            "--image_path",
            f"{IMAGES_PATH}",
            "--image_list_path",
            f"{IMAGE_LIST}",
            "--database_path",
            f"{DATASET_PATH}/database.db",
            "--output_path",
            f"{SPARSE_PATH}",
        ]
        call(mapper_command)
        # ---------------------------------------------

        # When done change status of model_i to done
        log_df.loc[log_df["Model"] == f"Model_{idx}", "status"] = 1
        log_df.to_csv(recon_log_path, index=False)
        print(current_model, " done. Changed status to 1")
