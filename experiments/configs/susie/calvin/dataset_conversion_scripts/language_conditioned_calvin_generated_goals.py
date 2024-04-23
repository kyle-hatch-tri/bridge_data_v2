"""
    This script processes the language annotated portions of the CALVIN dataset, writing it into TFRecord format.

    The dataset constructed with this script is meant to be used to train a language conditioned policy.

    Written by Pranav Atreya (pranavatreya@berkeley.edu).
"""

import numpy as np
import tensorflow as tf 
from tqdm import tqdm, trange
import os
from multiprocessing import Pool

from jax_diffusion_model import DiffusionModel

########## Dataset paths ###########
# raw_dataset_path = "/home/kylehatch/Desktop/hidql/data/calvin_data/task_ABCD_D"
raw_dataset_path = "/home/kylehatch/Desktop/hidql/calvin/dataset/task_ABCD_D"
tfrecord_dataset_path = "/home/kylehatch/Desktop/hidql/data/calvin_data_processed/language_conditioned_with_generated"


print("raw_dataset_path:", raw_dataset_path)

########## Main logic ###########
if not os.path.exists(tfrecord_dataset_path):
    os.mkdir(tfrecord_dataset_path)
if not os.path.exists(os.path.join(tfrecord_dataset_path, "training")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "training"))
if not os.path.exists(os.path.join(tfrecord_dataset_path, "validation")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "validation"))
if not os.path.exists(os.path.join(tfrecord_dataset_path, "training/A")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "training/A"))
if not os.path.exists(os.path.join(tfrecord_dataset_path, "training/B")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "training/B"))
if not os.path.exists(os.path.join(tfrecord_dataset_path, "training/C")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "training/C"))
if not os.path.exists(os.path.join(tfrecord_dataset_path, "training/D")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "training/D"))
if not os.path.exists(os.path.join(tfrecord_dataset_path, "validation/D")):
    os.mkdir(os.path.join(tfrecord_dataset_path, "validation/D"))

def make_seven_characters(id):
    id = str(id)
    while len(id) < 7:
        id = "0" + id
    return id

def tensor_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )

def string_to_feature(str_value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[str_value.encode("UTF-8")])
    )

def process_trajectory(function_data):
    global raw_dataset_path, tfrecord_dataset_path
    idx_range, letter, ctr, split, lang_ann = function_data
    unique_pid = split + "_" + letter + "_" + str(ctr)

    start_id, end_id = idx_range[0], idx_range[1]

    # We will filter the keys to only include what we need
    # Namely "rel_actions", "robot_obs", and "rgb_static"
    traj_rel_actions, traj_robot_obs, traj_rgb_static = [], [], []

    traj_rgb_generated_goals = []

    diffusion_model = DiffusionModel(50, num_samples=16)
    

    for ep_id in trange(start_id, end_id+1): # end_id is inclusive
        #print(unique_pid + ": iter " + str(ep_id-start_id) + " of " + str(end_id-start_id))

        ep_id = make_seven_characters(ep_id)
        timestep_data = np.load(os.path.join(raw_dataset_path, split, "episode_" + ep_id + ".npz"))
        
        rel_actions = timestep_data["rel_actions"]
        traj_rel_actions.append(rel_actions)

        robot_obs = timestep_data["robot_obs"]
        traj_robot_obs.append(robot_obs)

        rgb_static = timestep_data["rgb_static"] # not normalized, so we have to do normalization in another script
        traj_rgb_static.append(rgb_static)

        
        goal_images = diffusion_model.generate(lang_ann, rgb_static)
        traj_rgb_generated_goals.append(goal_images)
        import ipdb; ipdb.set_trace()
    
    traj_rel_actions, traj_robot_obs, traj_rgb_static = np.array(traj_rel_actions, dtype=np.float32), np.array(traj_robot_obs, dtype=np.float32), np.array(traj_rgb_static, dtype=np.uint8)
    traj_rgb_generated_goals = np.array(traj_rgb_generated_goals, dtype=np.uint8)

    # Determine the output path
    write_dir = os.path.join(tfrecord_dataset_path, split, letter)
    
    # Write the TFRecord
    output_tfrecord_path = os.path.join(write_dir, "traj" + str(ctr) + ".tfrecord")
    with tf.io.TFRecordWriter(output_tfrecord_path) as writer:
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "actions" : tensor_feature(traj_rel_actions),
                    "proprioceptive_states" : tensor_feature(traj_robot_obs),
                    "image_states" : tensor_feature(traj_rgb_static),
                    "language_annotation" : string_to_feature(lang_ann),
                    "generated_goals": tensor_feature(traj_rgb_generated_goals),
                }
            )
        )
        writer.write(example.SerializeToString())

    return end_id + 1 - start_id

# Let's prepare the inputs
function_inputs = []

# First let's do the train data
auto_lang_ann = np.load(os.path.join(raw_dataset_path, "training", "lang_annotations", "auto_lang_ann.npy"), allow_pickle=True)
auto_lang_ann = auto_lang_ann.item()
all_language_annotations = auto_lang_ann["language"]["ann"]
idx_ranges = auto_lang_ann["info"]["indx"]

scene_info = np.load(os.path.join(raw_dataset_path, "training", "scene_info.npy"), allow_pickle=True)
scene_info = scene_info.item()

A_ctr, B_ctr, C_ctr, D_ctr = 0, 0, 0, 0
for i, idx_range in enumerate(idx_ranges):
    start_idx = idx_range[0]
    if start_idx <= scene_info["calvin_scene_D"][1]:
        ctr = D_ctr
        D_ctr += 1
        letter = "D"
    elif start_idx <= scene_info["calvin_scene_B"][1]: # This is actually correct. In ascending order we have D, B, C, A
        ctr = B_ctr
        B_ctr += 1
        letter = "B"
    elif start_idx <= scene_info["calvin_scene_C"][1]:
        ctr = C_ctr
        C_ctr += 1
        letter = "C"
    else:
        ctr = A_ctr
        A_ctr += 1
        letter = "A"

    function_inputs.append((idx_range, letter, ctr, "training", all_language_annotations[i]))

# Next let's do the validation data
auto_lang_ann = np.load(os.path.join(raw_dataset_path, "validation", "lang_annotations", "auto_lang_ann.npy"), allow_pickle=True)
auto_lang_ann = auto_lang_ann.item()
all_language_annotations = auto_lang_ann["language"]["ann"]
idx_ranges = auto_lang_ann["info"]["indx"]

ctr = 0
for i, idx_range in enumerate(idx_ranges):
    function_inputs.append((idx_range, "D", ctr, "validation", all_language_annotations[i]))
    ctr += 1

traj_lens = []
print("Before process")
print("len(function_inputs):", len(function_inputs))
# Finally loop through and process everything
for function_input in tqdm(function_inputs, disable=True):
    traj_len = process_trajectory(function_input)
    traj_lens.append(traj_len)
print("After process")

print("np.sum(traj_lens):", np.sum(traj_lens))
print("np.mean(traj_lens):", np.mean(traj_lens))
print("len(traj_lens):", len(traj_lens))

# You can also parallelize execution with a process pool, see end of sister script


"""
export PYTHONPATH="/home/kylehatch/Desktop/hidql/bridge_data_v2/external/susie:$PYTHONPATH"
export DIFFUSION_MODEL_CHECKPOINT=/home/kylehatch/Desktop/hidql/susie-calvin-checkpoints/susie_test/test1_400smthlib_2024.02.21_06.44.06/40000/params_ema
python3 -u experiments/configs/susie/calvin/dataset_conversion_scripts/language_conditioned_calvin_generated_goals.py

"""