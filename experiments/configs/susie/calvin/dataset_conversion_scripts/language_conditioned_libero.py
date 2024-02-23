"""
    This script processes the language annotated portions of the CALVIN dataset, writing it into TFRecord format.

    The dataset constructed with this script is meant to be used to train a language conditioned policy.

    Written by Pranav Atreya (pranavatreya@berkeley.edu).
"""

import numpy as np
import tensorflow as tf 
from tqdm import tqdm 
import os
from multiprocessing import Pool
from glob import glob
import h5py
from libero.libero import benchmark
import cv2



########## Dataset paths ###########
render_videos = True 
raw_dataset_path = "/home/kylehatch/Desktop/hidql/data/libero_data"
tfrecord_dataset_path = "/home/kylehatch/Desktop/hidql/data/libero_data_processed"
# raw_dataset_path = "/home/kylehatch/Desktop/hidql/data/libero_data"
# tfrecord_dataset_path = "/home/kylehatch/Desktop/hidql/data/libero_data_processed"


print("raw_dataset_path:", raw_dataset_path)

########## Main logic ###########
os.makedirs(tfrecord_dataset_path, exist_ok=True)

def tensor_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()]))

def string_to_feature(str_value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str_value.encode("UTF-8")]))


def save_numpy_array_as_video(numpy_array, output_path, fps=30):
    # Ensure the array has the correct shape and dtype
    if numpy_array.shape[3] != 3 or numpy_array.dtype != np.uint8:
        raise ValueError("Invalid array shape or dtype. Expected shape (172, 128, 128, 3) and dtype np.uint8.")

    # Get video dimensions from the array shape
    height, width = numpy_array.shape[1], numpy_array.shape[2]

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can choose other codecs based on your requirements
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        # Loop through the frames and write them to the video
        for frame in numpy_array:
            video_writer.write(frame)

    except Exception as e:
        print(f"Error writing video frames: {e}")

    finally:
        # Release the VideoWriter object
        video_writer.release()


def batch_resize(images, new_h, new_w):
    (N,H,W,C) = images.shape
    # assert N*C <= 512

    instack = images.transpose((1,2,3,0)).reshape((H,W,C*N))

    outstack = cv2.resize(instack, (new_h, new_w))

    out_images = outstack.reshape((new_h,new_w,C,N)).transpose((3,0,1,2))

    return out_images


def sequential_resize(images, new_h, new_w):
    (N,H,W,C) = images.shape
    out_images = np.zeros((N, new_h, new_w, C), dtype=images.dtype)

    for i in range(N):
        out_images[i] = cv2.resize(images[i], (new_h, new_w))

    return out_images




trajectory_files = glob(os.path.join(raw_dataset_path, "**", "*.hdf5"), recursive=True)

benchmark_dict = benchmark.get_benchmark_dict()
traj_lens = []

for file_no, hdf5_file_path in enumerate(tqdm(trajectory_files, disable=True)):
    task_suite_name = hdf5_file_path.split("/")[-2]
    task_suite = benchmark_dict[task_suite_name]()
    # retrieve a specific task
    task_names = [task.name for task in task_suite.tasks]

    task_name = hdf5_file_path.split("/")[-1].split(".")[0]
    if "_demo" in task_name:
        task_name = task_name.split("_demo")[0]

    assert task_name in task_names, f"\"{task_name}\" not in task_names: {task_names}"

    
    task_id = task_names.index(task_name)
    task = task_suite.get_task(task_id)
    assert task.name == task_name, f"task.name: {task.name} != task_name: {task_name}"

    init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states

    task_description = task.language
    
    subdataset_name = hdf5_file_path.split("/")[-2]

    write_dir = os.path.join(tfrecord_dataset_path, subdataset_name, task_name)
    os.makedirs(write_dir, exist_ok=True)

    print(f"[{file_no + 1}/{len(trajectory_files)}] Processing file \"{hdf5_file_path}\"...")

    with h5py.File(hdf5_file_path, 'r') as hdf5_file:           
        demo_names = hdf5_file["data"].keys()
        assert len(demo_names) == init_states.shape[0], f"len(demo_names): {len(demo_names)}, init_states.shape: {init_states.shape}"

        get_demo_id = lambda x: int(x.split("_")[-1])
        demo_names = sorted(list(demo_names), key=get_demo_id)

        for i, demo in enumerate(tqdm(demo_names)): 
            agentview_rgb = hdf5_file["data"][demo]["obs"]["agentview_rgb"][:]
            eye_in_hand_rgb = hdf5_file["data"][demo]["obs"]["eye_in_hand_rgb"][:]

            agentview_rgb = np.flip(agentview_rgb, axis=1)
            eye_in_hand_rgb = np.flip(eye_in_hand_rgb, axis=1)

            joint_states = hdf5_file["data"][demo]["obs"]["joint_states"][:]
            gripper_states = hdf5_file["data"][demo]["obs"]["gripper_states"][:]
            ee_pos = hdf5_file["data"][demo]["obs"]["ee_pos"][:]
            ee_states = hdf5_file["data"][demo]["obs"]["ee_states"][:]
            ee_ori = hdf5_file["data"][demo]["obs"]["ee_ori"][:]
            proprioception = np.concatenate([joint_states, gripper_states, ee_pos, ee_states, ee_ori], axis=1)


            actions = hdf5_file["data"][demo]["actions"][:]
            rewards = hdf5_file["data"][demo]["rewards"][:]
            dones = hdf5_file["data"][demo]["dones"][:]
            states = hdf5_file["data"][demo]["states"]

            agentview_rgb = sequential_resize(agentview_rgb, 200, 200)
            eye_in_hand_rgb = sequential_resize(eye_in_hand_rgb, 200, 200)

            traj_lens.append(agentview_rgb.shape[0])
            
            demo_id = get_demo_id(demo)
            assert demo_id == i, f"demo_id: {demo_id}, i: {i}"

            if render_videos:
                video_array = np.concatenate([agentview_rgb, eye_in_hand_rgb], axis=2)
                video_array = video_array[..., ::-1]
                output_video_path = os.path.join("/home/kylehatch/Desktop/hidql/data/libero_data", "data_visualizations", subdataset_name, task_name, f"{demo}.avi")
                os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
                save_numpy_array_as_video(video_array, output_video_path, fps=30)

            # Write the TFRecord
            # output_tfrecord_path = os.path.join(write_dir, demo + ".tfrecord")
            output_tfrecord_path = os.path.join(write_dir, f"traj{demo_id}.tfrecord")
            
            with tf.io.TFRecordWriter(output_tfrecord_path) as writer:
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "actions" : tensor_feature(actions),
                            "proprioceptive_states" : tensor_feature(proprioception),
                            "image_states" : tensor_feature(agentview_rgb),
                            "language_annotation" : string_to_feature(task_description)
                        }
                    )
                )
                writer.write(example.SerializeToString())


print("np.sum(traj_lens):", np.sum(traj_lens))
print("np.mean(traj_lens):", np.mean(traj_lens))
print("len(traj_lens):", len(traj_lens))

# You can also parallelize execution with a process pool, see end of sister script



"""
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tensorflow 2.13.0 requires typing-extensions<4.6.0,>=3.6.6, but you have typing-extensions 4.9.0 which is incompatible.
tensorflow-probability 0.21.0 requires typing-extensions<4.6.0, but you have typing-extensions 4.9.0 which is incompatible.
Successfully installed filelock-3.13.1 jinja2-3.1.3 mpmath-1.3.0 networkx-3.2.1 nvidia-cublas-cu12-12.1.3.1 nvidia-cuda-cupti-cu12-12.1.105 nvidia-cuda-nvrtc-cu12-12.1.105 nvidia-cuda-runtime-cu12-12.1.105 nvidia-cudnn-cu12-8.9.2.26 nvidia-cufft-cu12-11.0.2.54 nvidia-curand-cu12-10.3.2.106 nvidia-cusolver-cu12-11.4.5.107 nvidia-cusparse-cu12-12.1.0.106 nvidia-nccl-cu12-2.19.3 nvidia-nvjitlink-cu12-12.3.101 nvidia-nvtx-cu12-12.1.105 sympy-1.12 torch-2.2.0 triton-2.2.0 typing-extensions-4.9.0
"""