import argparse
from pathlib import Path
from os import listdir
from PIL import Image
import math
import numpy as np
from sklearn.cluster import KMeans
import csv
import pandas as pd
import matplotlib.pyplot as plt


def get_gimball_yaw_pitch_roll(exif):
    """
    Reads yaw, pitch and roll angles from the exif data

    Input: exif data from image

    Output: [Yaw, Pitch, Roll] angles from the gimball in degrees
    """
    for k, v in exif.items():
        if k == 37500:

            v = str(v)
            splits = v.split("[")

            section = ""
            for _, v_spl in enumerate(splits):
                if "GimbalDegree(Y,P,R)" in v_spl:
                    section = v_spl
            split2 = section.split("\\")
            split3 = split2[0].split(":")
            gimball = split3[1].split(",")

            yaw = int(gimball[0])
            pitch = int(gimball[1])
            roll = int(gimball[2])

    return [yaw, pitch, roll]


# ------------------------------
# Conversion to LV95
# ------------------------------


def dd_mmss_ss_to_lat_lng(north, east):

    # calculate longitude:
    lat_degrees = north[0]
    lat_minutes = north[1]
    lat_seconds = north[2]
    lat_decimal_degrees = float(lat_degrees + (lat_minutes / 60) + (lat_seconds / 3600))

    lon_degrees = east[0]
    lon_minutes = east[1]
    lon_seconds = east[2]
    lon_decimal_degrees = float(lon_degrees + (lon_minutes / 60) + (lon_seconds / 3600))

    return lat_decimal_degrees, lon_decimal_degrees


# Convert sexagesimal angle (dd.mmss,ss) to seconds
def SexAngleToSeconds(dms):
    degree = 0
    minute = 0
    second = 0
    degree = math.floor(dms)
    minute = math.floor((dms - degree) * 100)
    second = (((dms - degree) * 100) - minute) * 100
    return second + (minute * 60) + (degree * 3600)


# Convert Decimal Degrees to Seconds of Arc (seconds only of D°M'S").
def DegToSec(angle):
    # Extract D°M'S".
    degree = int(angle)
    minute = int((angle - degree) * 100)
    second = (((angle - degree) * 100) - minute) * 100

    # Result in degrees sec (dd.mmss).
    return second + (minute * 60) + (degree * 3600)


# Convert decimal angle (° dec) to sexagesimal angle (dd.mmss,ss)
def DecToSexAngle(dec):
    degree = int(math.floor(dec))
    minute = int(math.floor((dec - degree) * 60))
    second = (((dec - degree) * 60) - minute) * 60
    return degree + (float(minute) / 100) + (second / 10000)


def WGStoLV95North(lat, lng):
    # Converts Decimal Degrees to Sexagesimal Degree.
    lat = DecToSexAngle(lat)
    lng = DecToSexAngle(lng)
    # Convert Decimal Degrees to Seconds of Arc.
    phi = DegToSec(lat)
    lda = DegToSec(lng)

    # Calculate the auxiliary values (differences of latitude and longitude
    # relative to Bern in the unit[10000"]).
    phi_aux = (phi - 169028.66) / 10000
    lda_aux = (lda - 26782.5) / 10000

    # Process Swiss (MN95) North calculation.
    north = (
        (1200147.07 + (308807.95 * phi_aux))
        + +(3745.25 * pow(lda_aux, 2))
        + +(76.63 * pow(phi_aux, 2))
        + -(194.56 * pow(lda_aux, 2) * phi_aux)
        + +(119.79 * pow(phi_aux, 3))
    )
    return north


def WGSToLV95East(lat, lng):
    # Converts Decimal Degrees to Sexagesimal Degree.
    lat = DecToSexAngle(lat)
    lng = DecToSexAngle(lng)
    # Convert Decimal Degrees to Seconds of Arc.
    phi = DegToSec(lat)
    lda = DegToSec(lng)

    # Calculate the auxiliary values (differences of latitude and longitude
    # relative to Bern in the unit[10000"]).
    phi_aux = (phi - 169028.66) / 10000
    lda_aux = (lda - 26782.5) / 10000

    # Process Swiss (MN95) East calculation.
    east = (
        (2600072.37 + (211455.93 * lda_aux))
        + -(10938.51 * lda_aux * phi_aux)
        + -(0.36 * lda_aux * pow(phi_aux, 2))
        + -(44.54 * pow(lda_aux, 3))
    )
    return east


def WGStoCHh(lat, lng, h):
    lat = DecToSexAngle(lat)
    lng = DecToSexAngle(lng)
    lat = SexAngleToSeconds(lat)
    lng = SexAngleToSeconds(lng)
    # Axiliary values (% Bern)
    lat_aux = (lat - 169028.66) / 10000
    lng_aux = (lng - 26782.5) / 10000
    h = (h - 49.55) + (2.73 * lng_aux) + (6.94 * lat_aux)
    return h


if __name__ == "__main__":

    # ------------------------------------------------------------------
    # -                           ARGPARSE                             -
    # ------------------------------------------------------------------

    # Argparse: Create parser
    parser = argparse.ArgumentParser()
    # Argparse: Add arguments
    parser.add_argument(dest="project_path", help="Path to the project folder")
    parser.add_argument(dest="images_path", help="Path to the images")
    parser.add_argument(
        dest="cluster_images",
        choices=["True", "False"],
        help="Indicate if provided images should be clustered into sub-models. True or False",
    )
    # optional
    parser.add_argument(
        "-presorted_im", "--presorted_im",
        help="Path to txt file which indicates which images should be used. List of image names",
        default=None
    )  
    parser.add_argument("-y_boundary", "--y_boundary", help="max_y: remove the images whose camera positions are outside the working area in the y direction")
    parser.add_argument("-x_boundary", "--x_boundary", help="max_x: remove the images whose camera positions are outside the working area in the x direction")
    parser.add_argument("-offset","--offset",
                        type=float, nargs=3, default=[0.0, 0.0, 0.0],
                        help="Offset to shift the initial point cloud coordinates to avoid excessively large coordinate values.") # This helps in maintaining numerical stability
    # Argparse: Parse arguments
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # -                           Set up                               -
    # ------------------------------------------------------------------

    # Paths:
    IMAGES_PATH = Path(f"{args.images_path}")
    PROJECT_PATH = Path(f"{args.project_path}")

    # If provided: read presorted images txt file to pandas dataframe
    if args.presorted_im is not None:
        IM_TO_USE_TXT = Path(f"{args.presorted_im}")
        df_use_im = pd.read_csv(IM_TO_USE_TXT, header=None, names=["im_names"])
        im_to_use_list = df_use_im["im_names"].tolist()

    # Create csv to io indicate batches and image positions/orientation in (shifted) LV95 (if not exist already)
    filename = Path(f"{PROJECT_PATH}/Images_LV95_batches.csv")
    if not filename.exists():
        add_to_project = False
        headers = [
            "x",
            "y",
            "z",
            "gimball_yaw",
            "gimball_pitch",
            "gimball_roll",
            "name",
            "batch",
        ]  # x, y, z in (shifted) LV95, gimball_yaw, gimball_pitch, gimball_roll angles in radians, name: name of image, batch: model batch it belonges to
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)
    else:
        add_to_project = True
        df_batches = pd.read_csv(filename)
        max_batch = df_batches["batch"].max()
        batch_numbering = int(max_batch)

    # ------------------------------------------------------------------
    # -           Get x,y,z,yaw,pitch,roll from EXIF-data              -
    # ------------------------------------------------------------------

    x = []  # east   (LV95 - constant)
    y = []  # north  (LV95 - constant)
    z = []  # height (LV95)
    gimball_yaw = []
    gimball_pitch = []
    gimball_roll = []
    name = []

    # Constants to shift coordinates (else numbers to big)
    offset = args.offset
    # print(offset)
    cx = offset[0]
    cy = offset[1]
    cz = offset[2]


    # Loop through data from images folder
    print("Getting EXIF data...")
    for image in listdir(IMAGES_PATH):
        if image.endswith(".JPG"):

            use_image = 1
            # Check if image should be used (if optional file is provided)
            if args.presorted_im is not None:
                if image not in im_to_use_list:
                    use_image = 0

            if use_image == 1:
                im = Image.open(IMAGES_PATH.joinpath(image))
                exif_data = im._getexif()

                wgs_north = None
                wgs_east = None
                for k, v in exif_data.items():
                    if k == 34853:
                        wgs_north = v.get(2)
                        wgs_east = v.get(4)
                        wgs_h = v.get(6)

                # Transform Coordinates to LV95
                lat_wgs, lng_wgs = dd_mmss_ss_to_lat_lng(wgs_north, wgs_east)

                LV95_north = WGStoLV95North(lat=lat_wgs, lng=lng_wgs)
                LV95_east = WGSToLV95East(lat=lat_wgs, lng=lng_wgs)
                LV95_h = WGStoCHh(lat=lat_wgs, lng=lng_wgs, h=wgs_h)

                # Shift values
                x_val = LV95_east - cx
                y_val = LV95_north - cy
                LV95_h = LV95_h - cz  

                gim_yaw, gim_pitch, gim_roll = get_gimball_yaw_pitch_roll(exif_data)

                # Convert to radians
                gim_yaw_rad = np.radians(gim_yaw)
                gim_pitch_rad = np.radians(gim_pitch)
                gim_roll_rad = np.radians(gim_roll)

                gimball_yaw.append(gim_yaw_rad)
                gimball_pitch.append(gim_pitch_rad)
                gimball_roll.append(gim_roll_rad)

                x.append(x_val)
                y.append(y_val)
                z.append(LV95_h)
                name.append(image)

    print("Done!")

    # Bounding the working area
    original_data = list(zip(x, y, z, gimball_yaw, gimball_pitch, gimball_roll, name))
    if args.y_boundary or args.x_boundary is not None:
        points = np.array([point[:3] for point in original_data])  # Extract x, y, z
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))

        trans_file = f"{PROJECT_PATH}/trans.txt"  # found using Cloudcompare - rotation/translation
        trans_matrix = np.loadtxt(trans_file)
        transformed_points = homogeneous_points.dot(trans_matrix.T)[:, :3]

        # fig = plt.figure()
        # plt.scatter(points[:, 0], points[:, 1], c='r', marker='o', label='Original')
        # # plt.scatter(transformed_points[:, 0], transformed_points[:, 1], c='b', marker='^', label='Transformed')
        # plt.xlabel('X-axis')
        # plt.ylabel('Y-axis')
        # plt.title('2D Visualization of Transformed Points')
        # plt.axis('equal')
        # plt.show()

        if args.y_boundary is not None:
            max_y = float(args.y_boundary)  # case1: 20; case2:16
        else:
            max_y = 1e9
        if args.x_boundary is not None:
            bound_x = float(args.x_boundary)
        else:
            bound_x = 1e9

        filtered_original_data = []
        for data, point in zip(original_data, transformed_points):
            if point[1] <= max_y or point[0] > bound_x:
                filtered_original_data.append(data)
    else:
        filtered_original_data = original_data

    # Create model batch label
    if args.cluster_images == "False":
        if add_to_project:
            lab = batch_numbering + 1
            labels = [lab] * len(name)
        else:
            labels = [0] * len(name)

    # ------------------------------------------------------------------
    # -                        Clustering                              -
    # ------------------------------------------------------------------
    if args.cluster_images == "True":
        print("Clustering data...")

        num_klusters = 4  # define number of klusters 10/8/6
        data = np.array(filtered_original_data)[:, :2]

        # Kmeans
        kmeans = KMeans(n_clusters=num_klusters, random_state=0)
        kmeans.fit(data)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        if add_to_project:
            labels = labels + (batch_numbering + 1)

        # # Plot the clustered data:
        # plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
        # plt.scatter(centers[:, 0], centers[:, 1], marker='*', c='r', s=200)
        # plt.title('K-means Clustering')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.show()

        print("Done!")

    # ------------------------------------------------------------------
    # -                      Save data to csv                          -
    # ------------------------------------------------------------------

    # csv_data = zip(x, y, z, gimball_yaw, gimball_pitch, gimball_roll, name, labels)
    csv_data = [original_data + (label,) for original_data, label in zip(filtered_original_data, labels)]

    with open(filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(csv_data)

    print(f"Added data to CSV file: '{filename}'")
