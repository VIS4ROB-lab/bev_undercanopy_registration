"""anormal scale detection"""

import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import os


# Function to read colmap info from a .txt file
def read_colmap_info(file_path):
    colmap_data = []
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 8:  # Ensuring the line has the expected number of parts
                filename, qw, qx, qy, qz, x, y, z = parts
                image_index = re.search(r"20240127_(\d{6})\.JPG", filename)
                if image_index:
                    image_index = image_index.group()
                    colmap_data.append((filename, float(x), float(y), float(z), image_index))
    return pd.DataFrame(colmap_data, columns=["image_name", "x", "y", "z", "image_index"])


# Function to read GPS info from a .csv file
def read_gps_info(file_path):
    gps_data = pd.read_csv(file_path)
    gps_data_filtered = gps_data[gps_data["batch"] == 0]
    extracted_data = []

    for index, row in gps_data_filtered.iterrows():
        name = row["name"]
        x = row["x"]  # Coordinates
        y = row["y"]
        z = row["z"]

        image_index_match = re.search(r"20240127_(\d{6})\.JPG", name)
        if image_index_match:
            image_index = image_index_match.group()
            extracted_data.append((name, float(x), float(y), float(z), image_index))

    gps_df = pd.DataFrame(extracted_data, columns=["image_name", "x", "y", "z", "image_index"])
    return gps_df


# Function to calculate distance between two points in 3D
def calculate_distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)


# Function to absolute differences between successive ratios
def segment_data_by_threshold(df, output_dir, threshold=5, min_length=10):
    differences = np.abs(np.diff(df["ratio"]))
    cut_points = np.where(differences > threshold)[0] + 1

    start = 0
    segment_count = 0
    segments = []
    for cut in cut_points:
        if (cut - start) >= min_length:
            save_segment_to_file(df.iloc[start:cut], segment_count, output_dir)
            segment_count += 1
            segments.append(df.iloc[start:cut])
        start = cut

    # last segment
    if start < len(df) and (len(df) - start) >= min_length:
        save_segment_to_file(df.iloc[start:], segment_count, output_dir)
        segments.append(df.iloc[start:])

    return segments


def save_segment_to_file(segment, segment_id, output_dir):
    # Formatting the file name
    file_path = os.path.join(output_dir, f"segment_{segment_id}.txt")
    with open(file_path, "w") as file:
        for _, row in segment.iterrows():
            image_name = row["image_index"].replace("20240127_", "andreas_park_20240127_")
            file.write(f"{image_name}\n")
    print(f"Segment saved: {file_path}")


# Main script to process the data and perform calculations
colmap_file_path = "/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/project_andreas_zigzagentire/Models/Model_0/shifted_LV95_model_txt/cam_extracted_m0.txt"  # 'cam_extracted_m{idx}.txt'extracted from colmap model
gps_file_path = "/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/project_andreas_zigzagentire/Images_LV95_batches.csv"  # 'Images_LV95_batches.csv' file path

colmap_df = read_colmap_info(colmap_file_path)
gps_df = read_gps_info(gps_file_path)

# Merge the two datasets on image_index
merged_df = pd.merge(colmap_df, gps_df, on="image_index", how="inner", suffixes=("_colmap", "_gps"))

# Calculate distances for every 3 images in sequence and their ratios
ratios = []
for i in range(4, len(merged_df) - 3, 3):  # Step every 3 images
    dist_colmap = calculate_distance(
        merged_df.iloc[i]["x_colmap"],
        merged_df.iloc[i]["y_colmap"],
        merged_df.iloc[i]["z_colmap"],
        merged_df.iloc[i + 3]["x_colmap"],
        merged_df.iloc[i + 3]["y_colmap"],
        merged_df.iloc[i + 3]["z_colmap"],
    )
    dist_gps = calculate_distance(
        merged_df.iloc[i]["x_gps"],
        merged_df.iloc[i]["y_gps"],
        merged_df.iloc[i]["z_gps"],
        merged_df.iloc[i + 3]["x_gps"],
        merged_df.iloc[i + 3]["y_gps"],
        merged_df.iloc[i + 3]["z_gps"],
    )
    if dist_gps > 0:  # Avoid division by zero
        ratios.append((merged_df.iloc[i]["image_index"], dist_colmap / dist_gps))

ratios_df = pd.DataFrame(ratios, columns=["image_index", "ratio"])

# Sort the DataFrame by numeric index to ensure the plot follows the sequence order
ratios_df.sort_values(by="image_index", inplace=True)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(ratios_df["image_index"], ratios_df["ratio"], marker="o", linestyle="-", color="blue")
plt.title("Ratio of Colmap Distance to GPS Distance per Image")
plt.xlabel("Image Index")
plt.ylabel("Ratio")
plt.grid(True)
# plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()

segment_path = "/home/zjw/SP/forest_3d_sarah_lia/kelly_forest_sp/case4/dataset_andreas_park/code_separate"
segments = segment_data_by_threshold(ratios_df, segment_path, threshold=5, min_length=10)
# 234,270
