import numpy as np
from transforms3d import quaternions
import math
import csv
import shutil
from pathlib import Path

# import pixloctuning.datasets.swisstopo as swisstopo
import swisstopo as swisstopo


# from easyidp MIT
def read_ccp(ccp_path):
    """Read ``*_calibrated_camera_parameters.txt`` for :class:`easyidp.reconstruct.Photo` object
    Parameters
    ----------
    ccp_path: str
        file path
    Returns
    -------
    dict
        .. code-block:: python
            img_configs = {
                'w': 4608,
                'h': 3456,
                'Image1.JPG': {
                    'cam_matrix':  array([[...]]),
                    'rad_distort': array([ 0.03833474, ...]),
                    'tan_distort': array([0.00240852, ...]),
                    'cam_pos':     array([ 21.54872207, ...]),
                    'cam_rot':     array([[ 0.78389904, ...]])},

                'Image2.JPG':
                    {...}
                }
    Notes
    -----
    It is the camera position info in local coordinate, the file looks like:
    .. code-block:: text
        fileName imageWidth imageHeight
        camera matrix K [3x3]
        radial distortion [3x1]
        tangential distortion [2x1]
        camera position t [3x1]
        camera rotation R [3x3]
        camera model m = K [R|-Rt] X
        DJI_0954.JPG 4608 3456
        3952.81247514184087776812 0 2233.46124792750424603582
        0 3952.81247514184087776812 1667.92521335858214115433
        0 0 1
        0.03833474118270804865 -0.01750917966495743258 0.02049798716391852335
        0.00240851666319534747 0.00292562392135245920
        21.54872206687879199194 -29.58734160676452162875 30.        85702810138878149360
        0.78389904231994589345 -0.62058396220897726892 -0.      01943803742353054573
        -0.62086105345046738169 -0.78318706257080084043 -0.     03390541741516269608
        0.00581753884797473961 0.03864674463298682638 -0.99923600083815289352
        DJI_0955.JPG ...
    Example
    -------
    Data prepare
    .. code-block:: python
        >>> import numpy as np
        >>> np.set_printoptions(suppress=True)
        >>> import easyidp as idp
        >>> test_data = idp.data.TestData()
        >>> param_folder = str(test_data.pix4d.maize_folder / "1_initial" / "params")
        >>> param = idp.pix4d.parse_p4d_param_folder(param_folder)
    Then use this function:
    .. code-block:: python
        >>> ccp = idp.pix4d.read_ccp(param['ccp'])
        >>> ccp.keys()
        dict_keys(['DJI_0954.JPG', 'w', 'h', 'DJI_0955.JPG', ... , 'DJI_0091.JPG', 'DJI_0092.JPG'])
        >>> ccp['w']
        4608
        >>> ccp['DJI_0954.JPG']
        {
            'cam_matrix':
                array([[3952.81247514,    0.        , 2233.46124793],
                       [   0.        , 3952.81247514, 1667.92521336],
                       [   0.        ,    0.        ,    1.        ]]),

            'rad_distort':
                array([ 0.03833474, -0.01750918,  0.02049799]),
            'tan_distort':
                array([0.00240852, 0.00292562]),
            'cam_pos':
                array([ 21.54872207, -29.58734161,  30.8570281 ]),
            'cam_rot':
                array([[ 0.78389904, -0.62058396, -0.01943804],
                       [-0.62086105, -0.78318706, -0.03390542],
                       [ 0.00581754,  0.03864674, -0.999236  ]])
        }
    """
    with open(ccp_path, "r") as f:
        """
        # for each block
        1   fileName imageWidth imageHeight
        2-4 camera matrix K [3x3]
        5   radial distortion [3x1]
        6   tangential distortion [2x1]
        7   camera position t [3x1]
        8-10   camera rotation R [3x3]
        """
        lines = f.readlines()

    img_configs = {}
    img_configs["images"] = []
    curr_image = None
    file_name = ""
    cam_mat_line1 = ""
    cam_mat_line2 = ""
    cam_rot_line1 = ""
    cam_rot_line2 = ""

    for i, line in enumerate(lines):
        if i < 8:
            pass
        else:
            block_id = (i - 7) % 10
            if block_id == 1:  # [line]: fileName imageWidth imageHeight
                if curr_image:
                    img_configs["images"].append(curr_image)
                curr_image = {}
                file_name, w, h = line[:-1].split()  # ignore \n character
                curr_image["file_name"] = file_name
                curr_image["w"] = int(w)
                curr_image["h"] = int(h)
            elif block_id == 2:
                cam_mat_line1 = np.fromstring(line[:-1], dtype=float, sep=" ")
            elif block_id == 3:
                cam_mat_line2 = np.fromstring(line, dtype=float, sep=" ")
            elif block_id == 4:
                cam_mat_line3 = np.fromstring(line, dtype=float, sep=" ")
                curr_image["cam_matrix"] = np.vstack([cam_mat_line1, cam_mat_line2, cam_mat_line3])
            elif block_id == 5:
                curr_image["rad_distort"] = np.fromstring(line, dtype=float, sep=" ")
            elif block_id == 6:
                curr_image["tan_distort"] = np.fromstring(line, dtype=float, sep=" ")
            elif block_id == 7:
                curr_image["cam_pos"] = np.fromstring(line, dtype=float, sep=" ")
            elif block_id == 8:
                cam_rot_line1 = np.fromstring(line, dtype=float, sep=" ")
            elif block_id == 9:
                cam_rot_line2 = np.fromstring(line, dtype=float, sep=" ")
            elif block_id == 0:
                cam_rot_line3 = np.fromstring(line, dtype=float, sep=" ")
                cam_rot = np.vstack([cam_rot_line1, cam_rot_line2, cam_rot_line3])
                curr_image["cam_rot"] = cam_rot

    return img_configs


# end easyipd code


def angle_to_rotmat1(omega, phi, kappa):
    C_B_b = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    cw = math.cos(omega)
    cp = math.cos(phi)
    ck = math.cos(kappa)
    sw = math.sin(omega)
    sp = math.sin(phi)
    sk = math.sin(kappa)

    R = np.dot(
        C_B_b,
        np.array(
            [
                [cp * ck, cw * sk + sw * sp * ck, sw * sk - cw * sp * ck],
                [-cp * sk, cw * ck - sw * sp * sk, sw * ck + cw * sp * sk],
                [sp, -sw * cp, cw * cp],
            ]
        ),
    )

    return R


def load_pix4d_sfm_all_param(file_cam_file, offset):
    cameras_dict = read_ccp(file_cam_file)

    if len(cameras_dict["images"]) == 0:
        return None

    first_cam = cameras_dict["images"][0]
    intrisic_param = {
        "w": first_cam["w"],
        "h": first_cam["h"],
        "f": first_cam["cam_matrix"][0, 0],
        "cx": first_cam["cam_matrix"][0, 2],
        "cy": first_cam["cam_matrix"][1, 2],
    }

    return cameras_dict["images"], intrisic_param


def load_pix4d_sfm_all_local_param(file_cam_file):
    cameras_dict = read_ccp(file_cam_file)

    if len(cameras_dict["images"]) == 0:
        return None

    # first_cam = cameras_dict['images'][0]
    # intrisic_param ={'w':first_cam['w'],'h':first_cam['h'],'f':first_cam['cam_matrix'][0,0],'cx':first_cam['cam_matrix'][0,2],'cy':first_cam['cam_matrix'][1,2]}
    points = []
    orientation = []
    filenames = []
    intrinsics = []

    for i, image in enumerate(cameras_dict["images"]):
        points.append([image["cam_pos"][0], image["cam_pos"][1], image["cam_pos"][2]])
        quat = quaternions.mat2quat(np.transpose(image["cam_rot"]))
        orientation.append([quat[0], quat[1], quat[2], quat[3]])  # w,x,y,z
        filenames.append(image["file_name"])
        intrinsics.append(
            {
                "w": image["w"],
                "h": image["h"],
                "f": image["cam_matrix"][0, 0],
                "cx": image["cam_matrix"][0, 2],
                "cy": image["cam_matrix"][1, 2],
            }
        )

    return points, orientation, filenames, intrinsics


# !!
def load_pix4d_sfm_model(file_extern_cam_file, image_names, offset):
    points = []
    orientations = []
    filenames = []
    with open(file_extern_cam_file, mode="r") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=" ")
        for row in csv_reader:
            if image_names is None or row["imageName"] in image_names:
                filenames.append(row["imageName"])
                points.append((float(row["X"]) - offset[0], float(row["Y"]) - offset[1], float(row["Z"]) - offset[2]))
                rotmat = angle_to_rotmat1(
                    math.radians(float(row["Omega"])),
                    math.radians(float(row["Phi"])),
                    math.radians(float(row["Kappa"])),
                )
                quat = quaternions.mat2quat(np.transpose(rotmat))
                orientations.append((quat[0], quat[1], quat[2], quat[3]))  # w, x, y, z

    return points, orientations, filenames


# !
def load_pix4d_gps_model(file_gps_file, image_names, offset):  # using the swiss coord system lv95
    points = []
    orientations = []
    filenames = []
    converter = swisstopo.GPSConverter()
    with open(file_gps_file, mode="r") as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=" ")
        for row in csv_reader:
            if image_names is None or row["imageName"] in image_names:
                filenames.append(row["imageName"])

                chh = converter.WGStoCHh(row["latitude"], row["longitude"], row["altitude"])
                che = converter.WGSToLV95East(row["latitude"], row["longitude"])
                chn = converter.WGStoLV95North(row["latitude"], row["longitude"])

                points.append((float(che) - offset[0], float(chn) - offset[1], float(chh) - offset[2]))
                rotmat = angle_to_rotmat1(
                    math.radians(float(row["Omega"])),
                    math.radians(float(row["Phi"])),
                    math.radians(float(row["Kappa"])),
                )
                quat = quaternions.mat2quat(np.transpose(rotmat))
                orientations.append((quat[0], quat[1], quat[2], quat[3]))  # w, x, y, z

    return points, orientations, filenames


import copy


def convert_pix4d_gps_to_lv95(file_gps_file, file_lv95_file):

    converter = swisstopo.GPSConverter()

    with open(file_gps_file, mode="r") as in_csv_file:
        with open(file_lv95_file, mode="w") as out_csv_file:
            csv_reader = csv.DictReader(in_csv_file, delimiter=" ")
            new_filenames = copy.deepcopy(csv_reader.fieldnames)
            new_filenames[1] = "X"
            new_filenames[2] = "Y"
            new_filenames[3] = "Z"
            csv_writer = csv.DictWriter(out_csv_file, fieldnames=new_filenames, delimiter=" ")
            csv_writer.writeheader()
            print(csv_reader.fieldnames)
            for row in csv_reader:
                row["X"] = f"{converter.WGSToLV95East(float(row['latitude']), float(row['longitude'])):.6f}"
                row["Y"] = f"{converter.WGStoLV95North(float(row['latitude']), float(row['longitude'])):.6f}"
                row["Z"] = (
                    f"{converter.WGStoCHh(float(row['latitude']), float(row['longitude']), float(row['altitude'])):.6f}"
                )
                del row["latitude"], row["longitude"], row["altitude"]
                csv_writer.writerow(row)


def copy_rename_images_pix4d(
    curr_file_extern_cam_file, new_file_extern_cam_file, current_image_dir, new_image_dir, filename_prefix
):
    in_img_dir = Path(current_image_dir)
    out_img_dir = Path(new_image_dir)
    out_img_dir.mkdir(parents=True)
    with open(curr_file_extern_cam_file, mode="r") as in_csv_file:
        with open(new_file_extern_cam_file, mode="w") as out_csv_file:
            csv_reader = csv.DictReader(in_csv_file, delimiter=" ")
            csv_writer = csv.DictWriter(out_csv_file, fieldnames=csv_reader.fieldnames, delimiter=" ")
            csv_writer.writeheader()
            for row in csv_reader:
                file_new_name = filename_prefix + row["imageName"]
                shutil.copyfile(in_img_dir / row["imageName"], out_img_dir / file_new_name)
                row["imageName"] = file_new_name
                csv_writer.writerow(row)


def rename_images_pix4d(input_file, output_file, filename_prefix):
    with open(input_file, mode="r") as in_csv_file:
        with open(output_file, mode="w") as out_csv_file:
            csv_reader = csv.DictReader(in_csv_file, delimiter=" ")
            csv_writer = csv.DictWriter(out_csv_file, fieldnames=csv_reader.fieldnames, delimiter=" ")
            csv_writer.writeheader()
            for row in csv_reader:
                file_new_name = filename_prefix + row["imageName"]
                row["imageName"] = file_new_name
                csv_writer.writerow(row)
