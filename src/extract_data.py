import cv2
import rosbag
import rospy
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import datetime
import os
import glob
import yaml
import json
import shutil
import sys
import pdb
from transformations import *
import argparse

cv_bridge = CvBridge()


def write_imgs(imgs, imgfolders, frame_id):
    cvimgs = []
    for idx, im in enumerate(imgs):
        try:
            cvimgs.append(cv_bridge.imgmsg_to_cv2(im, desired_encoding="bgr8"))
        except CvBridgeError as e:
            print(e)
            exit(0)
    filename = "frame_%04d.jpg" % frame_id
    print(filename)
    for idx, im in enumerate(cvimgs):
        cv2.imwrite(os.path.join(imgfolders[idx], filename), im)


def write_depth_img(img, imgfolder, frame_id):
    cvimg = None
    try:
        cvimg = cv_bridge.imgmsg_to_cv2(img, desired_encoding="16UC1")
    except CvBridgeError as e:
        print(e)
        exit(0)

    filename = "frame_%04d.npy" % frame_id
    np.save(os.path.join(imgfolder, filename), cvimg)


def get_transform_matrix(transform):
    tft = transform.transform.translation
    tf_trans_list = [tft.x, tft.y, tft.z]
    tfr = transform.transform.rotation
    tf_rot_list = [tfr.x, tfr.y, tfr.z, tfr.w]
    mat = fromTranslationRotation(tf_trans_list, tf_rot_list)
    return mat


def extract_data(topics, filename, jsonfile, base_data_folder, args):
    basefile = os.path.splitext(os.path.basename(filename))[0]
    basefolder = os.path.split(os.path.dirname(os.path.abspath(filename)))[-1]
    data_folder = os.path.join(base_data_folder, basefile)

    if not os.path.exists(data_folder):
        os.mkdir(data_folder)

    print(data_folder)
    image_folders = ["rgb"]
    shutil.copy(jsonfile, data_folder)

    full_image_folders = []

    for imgf in image_folders:
        img_folder = os.path.join(data_folder, imgf)
        full_image_folders.append(img_folder)
        if not os.path.exists(img_folder):
            print(img_folder)
            os.mkdir(img_folder)

    depthimgfolder = os.path.join(data_folder, "depth")
    if not os.path.exists(depthimgfolder):
        os.mkdir(depthimgfolder)

    bag = rosbag.Bag(filename)
    hdimg = None
    depthimg = None
    wrench = None
    times = []
    joint_state_positions = None
    joint_state_velocities = None
    joint_state_effort = None
    joint_names = None
    base_velocity = None
    gripper_command = None
    min_time_duration = rospy.Duration.from_sec(1 / 35.0)
    last_time_stamp = rospy.Time(0)
    previous_joint_time = None
    previous_joint_states = None
    predicted_joint_positions = None
    camera_matrix = None
    mat1 = None
    mat2 = None
    mat3 = None
    mat4 = None
    mat4 = np.array(
        [
            [4.89630558e-12, 4.89641661e-12, 1.00000000e00, -7.97960000e-02],
            [-1.00000000e00, -2.22044605e-16, 4.89641661e-12, 2.20000000e-02],
            [0.00000000e00, -1.00000000e00, 4.89630558e-12, 2.15172000e-01],
            [0.00000000e00, 0.00000000e00, 0.00000000e00, 1.00000000e00],
        ]
    )

    # since there is at least one bagfile with no tf_static
    palm_wrist_roll_translation = [0.012, 0.0, 0.1405]
    palm_wrist_roll_rotation = [0.0, 0.0, 1.0, -1.03411553555e-13]
    palm_wrist_roll = fromTranslationRotation(
        palm_wrist_roll_translation, palm_wrist_roll_rotation
    )
    # palm_wrist_roll = None
    wrist_roll_flex = None
    wrist_flex_arm_roll = None
    arm_roll_flex = None
    arm_flex_lift = None
    arm_lift_base = None
    palm_position = None
    wrist_flex_to_sensor_mnt = None
    wrist_sensor_frame = None
    wrist_sensor_frame_to_roll = None

    frame_id = 0
    cnt = 0
    topics_to_read = topics.values()
    arm_goals = 0
    for topic, msg, t in bag.read_messages(topics=topics_to_read):
        if topic == topics["rgb"]:
            hdimg = msg
        if topic == topics["depth"]:
            depthimg = msg
        if topic == topics["wrench"]:
            f = [t.to_sec(), msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z]
            f.extend([msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z])
            f = np.array(f)
            if wrench is None:
                wrench = f[np.newaxis, :]
            else:
                wrench = np.vstack((wrench, f[np.newaxis, :]))
        elif (
            topic == topics["joint_states"]
            and mat1 is not None
            and mat2 is not None
            and mat3 is not None
            and mat4 is not None
        ):
            positions = [t.to_sec()]
            velocities = [t.to_sec()]
            effort = [t.to_sec()]
            positions.extend(msg.position)
            velocities.extend(msg.velocity)
            effort.extend(msg.effort)
            positions = np.array(positions)
            velocities = np.array(velocities)
            effort = np.array(effort)
            if joint_state_positions is None:
                joint_state_positions = positions[np.newaxis, :]
            else:
                joint_state_positions = np.vstack(
                    (joint_state_positions, positions[np.newaxis, :])
                )

            if joint_state_velocities is None:
                joint_state_velocities = velocities[np.newaxis, :]
            else:
                joint_state_velocities = np.vstack(
                    (joint_state_velocities, velocities[np.newaxis, :])
                )

            if joint_state_effort is None:
                joint_state_effort = effort[np.newaxis, :]
            else:
                joint_state_effort = np.vstack(
                    (joint_state_effort, effort[np.newaxis, :])
                )
            if joint_names is None:
                joint_names = msg.name
                print(joint_names)

            cam_matrix = mat1.dot(mat2).dot(mat3).dot(mat4)
            cam_translation = translation_from_matrix(cam_matrix)
            cam_rotation = cam_matrix.copy()
            cam_rotation[:, 3] = 0.0
            cam_rotation[3, :] = 0.0
            cam_rotation[3, 3] = 1.0
            q = quaternion_from_matrix(cam_rotation)
            euler = euler_from_quaternion(q)
            # in pyrender, z-axis of camera points away from scene
            q = quaternion_from_euler(euler[0] + np.pi, euler[1], euler[2])
            cam_matrix = fromTranslationRotation(cam_translation, q)

            if camera_matrix is None:
                camera_matrix = cam_matrix[np.newaxis, :, :]
            else:
                camera_matrix = np.vstack((camera_matrix, cam_matrix[np.newaxis, :, :]))
        elif topic == topics["tf"] or topic == topics["tf_static"]:
            cnt += 1
            for transform in msg.transforms:
                if (
                    transform.header.frame_id == "base_link"
                    and transform.child_frame_id == "torso_lift_link"
                ):
                    mat1 = get_transform_matrix(transform)
                    # print(mat1[2][3])
                elif (
                    transform.header.frame_id == "torso_lift_link"
                    and transform.child_frame_id == "head_pan_link"
                ):
                    mat2 = get_transform_matrix(transform)
                elif (
                    transform.header.frame_id == "head_pan_link"
                    and transform.child_frame_id == "head_tilt_link"
                ):
                    mat3 = get_transform_matrix(transform)
                elif (
                    transform.header.frame_id == "head_tilt_link"
                    and transform.child_frame_id == "head_rgbd_sensor_link"
                ):
                    mat4 = get_transform_matrix(transform)
                elif (
                    transform.header.frame_id == "base_link"
                    and transform.child_frame_id == "arm_lift_link"
                ):
                    arm_lift_base = get_transform_matrix(transform)
                elif (
                    transform.header.frame_id == "arm_lift_link"
                    and transform.child_frame_id == "arm_flex_link"
                ):
                    arm_flex_lift = get_transform_matrix(transform)
                elif (
                    transform.header.frame_id == "arm_flex_link"
                    and transform.child_frame_id == "arm_roll_link"
                ):
                    arm_roll_flex = get_transform_matrix(transform)
                elif (
                    transform.header.frame_id == "arm_roll_link"
                    and transform.child_frame_id == "wrist_flex_link"
                ):
                    wrist_flex_arm_roll = get_transform_matrix(transform)
                elif (
                    transform.header.frame_id == "wrist_flex_link"
                    and transform.child_frame_id == "wrist_ft_sensor_mount_link"
                ):
                    wrist_flex_to_sensor_mnt = get_transform_matrix(transform)
                elif (
                    transform.header.frame_id == "wrist_ft_sensor_mount_link"
                    and transform.child_frame_id == "wrist_ft_sensor_frame"
                ):
                    wrist_sensor_frame = get_transform_matrix(transform)
                elif (
                    transform.header.frame_id == "wrist_ft_sensor_frame"
                    and transform.child_frame_id == "wrist_roll_link"
                ):
                    wrist_sensor_frame_to_roll = get_transform_matrix(transform)
                elif (
                    transform.header.frame_id == "wrist_roll_link"
                    and transform.child_frame_id == "hand_palm_link"
                ):
                    palm_wrist_roll = get_transform_matrix(transform)

        if all(
            var is not None
            for var in [
                mat1,
                mat2,
                mat3,
                mat4,
                arm_lift_base,
                arm_flex_lift,
                arm_roll_flex,
                wrist_flex_arm_roll,
                wrist_flex_to_sensor_mnt,
                wrist_sensor_frame,
                wrist_sensor_frame_to_roll,
            ]
        ):
            base_to_camera = mat1.dot(mat2).dot(mat3).dot(mat4)
            base_to_palm = (
                arm_lift_base.dot(arm_flex_lift)
                .dot(arm_roll_flex)
                .dot(wrist_flex_arm_roll)
                .dot(wrist_flex_to_sensor_mnt)
                .dot(wrist_sensor_frame)
                .dot(wrist_sensor_frame_to_roll)
                .dot(palm_wrist_roll)
            )
            camera_to_palm = np.linalg.inv(base_to_camera).dot(base_to_palm)
            camera_to_palm = camera_to_palm[:3, 3]
            camera_to_palm = np.hstack((t.to_sec(), camera_to_palm))
            if palm_position is None:
                palm_position = camera_to_palm[np.newaxis, :]
            else:
                palm_position = np.vstack(
                    (palm_position, camera_to_palm[np.newaxis, :])
                )
        if hdimg and depthimg:
            if (t - last_time_stamp) > min_time_duration:
                write_imgs([hdimg], full_image_folders, frame_id)
                write_depth_img(depthimg, depthimgfolder, frame_id)
                last_time_stamp = t
                frame_id += 1
                times.append(t.to_sec())

    filename = os.path.join(data_folder, "palm_positions_full")
    np.save(filename, palm_position, allow_pickle=False)

    #####
    times = np.array(times)
    filename = os.path.join(data_folder, "image_timestamps")
    np.save(filename, times, allow_pickle=False)

    filename = os.path.join(data_folder, "wrench_full")
    np.save(filename, wrench, allow_pickle=False)

    filename = os.path.join(data_folder, "joint_names")
    np.save(filename, joint_names)

    filename = os.path.join(data_folder, "joint_state_positions_full")
    np.save(filename, joint_state_positions, allow_pickle=False)

    filename = os.path.join(data_folder, "joint_state_velocities_full")
    np.save(filename, joint_state_velocities, allow_pickle=False)

    filename = os.path.join(data_folder, "joint_state_efforts_full")
    np.save(filename, joint_state_effort, allow_pickle=False)

    filename = os.path.join(data_folder, "camera_matrix_full")
    np.save(filename, camera_matrix, allow_pickle=False)

    return data_folder


def resample(data_folder, args):
    image_ts = np.load(os.path.join(data_folder, "image_timestamps.npy"))

    original_files = [
        "joint_state_efforts_full.npy",
        "joint_state_positions_full.npy",
        "joint_state_velocities_full.npy",
        "wrench_full.npy",
        "camera_matrix_full.npy",
        "palm_positions_full.npy",
    ]
    resampled_files = [
        "joint_state_efforts.npy",
        "joint_state_positions.npy",
        "joint_state_velocities.npy",
        "wrench.npy",
        "camera_matrix.npy",
        "palm_positions.npy",
    ]
    camera_matrix_ids = None
    for tt, tt2 in zip(original_files, resampled_files):
        if not os.path.exists(os.path.join(data_folder, tt)):
            print(os.path.join(data_folder, tt))
            continue
        try:
            wrench = np.load(os.path.join(data_folder, tt))
            wrench_resampled = os.path.join(data_folder, tt2)

        except:
            pdb.set_trace()

        # pad beginning and end with zero velocity
        if "velocity" in tt or "velocities" in tt:
            wrench = np.vstack((wrench[0], wrench))
            wrench[0][1:] = 0.0  # set first row to 0 (except timestamp)
            wrench = np.vstack((wrench, wrench[-1]))
            wrench[-1][1:] = 0.0
        elif "matrix" in tt:
            wrench = np.vstack((wrench[0][np.newaxis, :, :], wrench))
            wrench = np.vstack((wrench, wrench[-1][np.newaxis, :, :]))
        else:
            wrench = np.vstack((wrench[0], wrench))
            wrench = np.vstack((wrench, wrench[-1]))
        if not "matrix" in tt:
            # find first entry whose timestamp is greater than image ts
            # and subtract 1
            ids = [np.argmax(wrench[:, 0] > ts) - 1 for ts in image_ts]
            # pad the beginning
            for idx, i in enumerate(ids):
                # this means there is no timestamp in data > image_ts
                # so we use the first datapoint
                if i == -1:
                    ids[idx] = 0
                else:
                    break
            # pad the end
            # loop backwards
            for idx in range(len(ids) - 1, -1, -1):
                # no ts found > image_ts
                # so pad with last known value
                if ids[idx] == -1:
                    ids[idx] = wrench.shape[0] - 1
                else:
                    break
        else:
            ids = camera_matrix_ids

        if "joint_state_positions_full" in tt:
            camera_matrix_ids = ids
        wrench_rs = wrench[ids]
        np.save(wrench_resampled, wrench_rs, allow_pickle=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bagfile_path", help="Example:/path/to/bagfiles/folder")
    parser.add_argument("base_data_folder", help="Example:/path/to/store/data")
    parser.add_argument(
        "-b", "--include_base", action="store_true", help="generate base velocities"
    )
    parser.add_argument(
        "-s",
        "--single_bagfile",
        type=str,
        default=None,
        help="specify filename of single bagfile in bagfile_path",
    )
    parser.add_argument(
        "-j",
        "--single_jsonfile",
        type=str,
        default=None,
        help="specify filename of single json file in bagfile_path",
    )
    args = parser.parse_args()

    path = args.bagfile_path
    base_data_folder = args.base_data_folder
    topics = {
        "rgb": "/hsrb/head_rgbd_sensor/rgb/image_raw",
        "wrench": "/hsrb/wrist_wrench/compensated",
        "joint_states": "/hsrb/robot_state/joint_states",
        #'cmd_vel': '/hsrb/command_velocity',
        #'gripper': '/hsrb/gripper_controller/command',
        "tf": "/tf",
        "tf_static": "/tf_static",
        "depth": "/hsrb/head_rgbd_sensor/depth_registered/image_raw",
    }

    if args.single_bagfile is None:
        files = glob.glob(path + "/*.bag")
        files = sorted(files)
        json_files = glob.glob(path + "/*.json")
        json_files = sorted(json_files)
    else:
        files = [os.path.join(path, args.single_bagfile)]
        json_files = [os.path.join(path, args.single_jsonfile)]

    print(json_files)
    print("Found %d bagfiles and %d json files" % (len(files), len(json_files)))
    for bagfile, jsonfile in zip(files, json_files):
        basefile = os.path.splitext(os.path.basename(bagfile))[0]
        basefolder = os.path.split(os.path.dirname(os.path.abspath(bagfile)))[-1]
        data_folder = extract_data(topics, bagfile, jsonfile, base_data_folder, args)
        resample(data_folder, args)


if __name__ == "__main__":
    main()
