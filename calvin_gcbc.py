import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tqdm
from absl import app, flags, logging
from flax.training import checkpoints
from ml_collections import config_flags
import yaml

from jaxrl_m.data.text_processing import text_processors

from jaxrl_m.agents import agents
from jaxrl_m.common.common import shard_batch
from jaxrl_m.common.wandb import WandBLogger
from jaxrl_m.data.calvin_dataset import CalvinDataset, glob_to_path_list
from jaxrl_m.utils.timer_utils import Timer
from jaxrl_m.vision import encoders

from s3_save import S3SyncCallback

try:
    from jax_smi import initialise_tracking  # type: ignore

    initialise_tracking()
except ImportError:
    pass

FLAGS = flags.FLAGS

# flags.DEFINE_string("name", "", "Experiment name.")
flags.DEFINE_string("wandb_proj_name", "jaxrl_m_calvin_gcbc", "Experiment name.")
flags.DEFINE_string("s3_save_uri", "", "Experiment name.")
flags.DEFINE_integer("debug", 0, "Debug config")
flags.DEFINE_integer("save_to_s3", 1, "Debug config")
flags.DEFINE_integer("seed", None, "Debug config")
flags.DEFINE_integer("log_to_wandb", 1, "Debug config")

# flags.DEFINE_string("dataset", "", "Experiment name.")
flags.DEFINE_string("algo", "", "Experiment name.")
flags.DEFINE_string("description", "", "Experiment name.")

config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

config_flags.DEFINE_config_file(
    "calvin_dataset_config",
    None,
    "File path to the CALVIN dataset configuration.",
    lock_config=False,
)


def save_dict_as_yaml(savepath, data):
    with open(savepath, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

import cv2
def save_video(output_video_file, frames):
     # Extract frame dimensions
    height, width, _ = frames.shape[1:]

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also use other codecs such as 'XVID'
    fps = 30  # Adjust the frame rate as needed

    os.makedirs(os.path.dirname(output_video_file), exist_ok=True)
    video_writer = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))

    # Write each frame to the video file
    for frame in frames:
        bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(bgr_frame)

    # Release the video writer object
    video_writer.release()

def main(_):
    devices = jax.local_devices()
    num_devices = len(devices)

    print("os.getenv(\"CUDA_VISIBLE_DEVICES\"):", os.getenv("CUDA_VISIBLE_DEVICES"))
    print("devices:", devices)
    print("num_devices:", num_devices)

    if FLAGS.seed is not None:
        FLAGS.config.seed = FLAGS.seed

    if FLAGS.debug:
        FLAGS.config.batch_size = 24
        FLAGS.config.num_val_batches = 2
        FLAGS.config.num_steps = 100
        FLAGS.config.log_interval = 20
        FLAGS.config.eval_interval = 90
        FLAGS.config.save_interval = 80

        # data_path_list = FLAGS.config.data_path.split("/")
        # if "calvin" in FLAGS.config.data_path:
        #     data_path_list[data_path_list.index("calvin_data_processed")] += "_subset"
        # FLAGS.config.data_path = "/".join(data_path_list)

        FLAGS.wandb_proj_name = "el_trasho"

    
    assert FLAGS.config.batch_size % num_devices == 0

    # Only differences are how I handle the batch shardings 
    # sharding = jax.sharding.PositionalSharding(devices) ###$$$###
    # shard_fn = partial(shard_batch, sharding=sharding)

    # prevent tensorflow from using GPUs
    tf.config.set_visible_devices([], "GPU")

    # set up wandb and logging
    wandb_config = WandBLogger.get_default_config()
    wandb_config.update(
        {
            "project": FLAGS.wandb_proj_name + f"_{FLAGS.config.dataset_name}",
            # "exp_descriptor": FLAGS.name,
            "exp_descriptor": f"{FLAGS.algo}_{FLAGS.description}",
            "seed":FLAGS.config.seed,
        }
    )

    
    wandb_logger = WandBLogger(
        wandb_config=wandb_config,
        variant=FLAGS.config.to_dict(),
        # debug=FLAGS.debug,
        debug=not FLAGS.log_to_wandb,
    )

    # save_dir = tf.io.gfile.join(FLAGS.config.save_dir, wandb_logger.config.project, f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}",)
    # s3_sync_callback = S3SyncCallback(os.path.abspath(save_dir), FLAGS.s3_save_uri + "/" + wandb_logger.config.project + "/" + f"{wandb_logger.config.exp_descriptor}_{wandb_logger.config.unique_identifier}")
    save_dir = tf.io.gfile.join(FLAGS.config.save_dir, FLAGS.wandb_proj_name, f"{FLAGS.config.dataset_name}", f"{FLAGS.algo}", f"{FLAGS.description}", f"seed_{FLAGS.config.seed}", f"{wandb_logger.config.unique_identifier}")
    os.makedirs(save_dir, exist_ok=True)
    s3_sync_callback = S3SyncCallback(os.path.abspath(save_dir), os.path.join(FLAGS.s3_save_uri, FLAGS.wandb_proj_name, f"{FLAGS.config.dataset_name}", f"{FLAGS.algo}", f"{FLAGS.description}", f"seed_{FLAGS.config.seed}", f"{wandb_logger.config.unique_identifier}"))  
    print("save_dir:", save_dir)
    print("s3_sync_callback.s3_uri:", s3_sync_callback.s3_uri)
    print("FLAGS.seed:", FLAGS.seed)
    print("FLAGS.config.seed:", FLAGS.config.seed)
    print("wandb_logger.config.project:", wandb_logger.config.project)
    print("wandb_logger.config.exp_descriptor:", wandb_logger.config.exp_descriptor)

    save_dict_as_yaml(os.path.join(save_dir, "config.yaml"), FLAGS.config.to_dict())
    if FLAGS.save_to_s3:
        s3_sync_callback.upload_base_savedir()

    print("FLAGS.config.data_path:", FLAGS.config.data_path)

    # load datasets
    assert type(FLAGS.calvin_dataset_config.include[0]) == list
    task_paths = [
        glob_to_path_list(
            path, prefix=FLAGS.config.data_path, exclude=FLAGS.calvin_dataset_config.exclude
        )
        for path in FLAGS.calvin_dataset_config.include
    ]

    train_paths = [task_paths[0]]
    val_paths = [task_paths[1]]

    obs_horizon = FLAGS.config.get("obs_horizon")

    print("train_paths[0][:25]:", train_paths[0][:25])
    print("len(train_paths[0]):", len(train_paths[0]))
    print("val_paths[0][:25]:", val_paths[0][:25])
    print("len(val_paths[0]):", len(val_paths[0]))

    train_data = CalvinDataset(
        train_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        num_devices=num_devices, ### think num_devices is just an extra kwargs???
        train=True,
        action_proprio_metadata=FLAGS.calvin_dataset_config.action_proprio_metadata,
        sample_weights=FLAGS.calvin_dataset_config.sample_weights,
        **FLAGS.config.dataset_kwargs,
    )
    val_data = CalvinDataset(
        val_paths,
        FLAGS.config.seed,
        batch_size=FLAGS.config.batch_size,
        action_proprio_metadata=FLAGS.calvin_dataset_config.action_proprio_metadata,
        train=False,
        **FLAGS.config.dataset_kwargs,
    )

    def process_text(batch): 
        if text_processor is None:
            batch["goals"].pop("language")
        else:
            batch["goals"]["language"] = text_processor.encode(
                #[s.decode("utf-8") for s in batch["goals"]["language"]]
                [s for s in batch["goals"]["language"]]
            )
        return batch

    if FLAGS.config.language_conditioned:
        assert FLAGS.config.encoder == "resnetv1-34-bridge-film", f"FLAGS.config.encoder: {FLAGS.config.encoder}"
        text_processor = text_processors[FLAGS.config.text_processor](**FLAGS.config.text_processor_kwargs)
        ###$$$###
        # train_data_iter = map(shard_fn, map(process_text, train_data.tf_dataset.as_numpy_iterator()))
        train_data_iter = map(process_text, train_data.tf_dataset.as_numpy_iterator())
    else:
        # assert FLAGS.config.text_processor is None, f"FLAGS.config.text_processor: {FLAGS.config.text_processor}"
        assert FLAGS.config.encoder == "resnetv1-34-bridge", f"FLAGS.config.encoder: {FLAGS.config.encoder}"
        train_data_iter = train_data.iterator()

        
        

    print('FLAGS.config.dataset_kwargs.goal_relabeling_strategy:', FLAGS.config.dataset_kwargs.goal_relabeling_strategy)
    print('FLAGS.config.dataset_kwargs.goal_relabeling_kwargs:', FLAGS.config.dataset_kwargs.goal_relabeling_kwargs)


    example_batch = next(train_data_iter)
    logging.info(f"Batch size: {example_batch['observations']['image'].shape[0]}")
    logging.info(f"Number of devices: {num_devices}")
    logging.info(
        f"Batch size per device: {example_batch['observations']['image'].shape[0] // num_devices}"
    )


    # we shard the leading dimension (batch dimension) accross all devices evenly
    sharding = jax.sharding.PositionalSharding(devices)
    example_batch = shard_batch(example_batch, sharding) ###$$$###

    # define encoder
    encoder_def = encoders[FLAGS.config.encoder](**FLAGS.config.encoder_kwargs)

    # initialize agent
    rng = jax.random.PRNGKey(FLAGS.config.seed)
    rng, construct_rng = jax.random.split(rng)
    agent = agents[FLAGS.config.agent].create(
        rng=construct_rng,
        observations=example_batch["observations"],
        goals=example_batch["goals"],
        actions=example_batch["actions"],
        encoder_def=encoder_def,
        **FLAGS.config.agent_kwargs,
    )

    print("type(agent):", type(agent))

    print("FLAGS.config.encoder:", FLAGS.config.encoder)

    if FLAGS.config.resume_path is not None:
        agent = checkpoints.restore_checkpoint(FLAGS.config.resume_path, target=agent)
    # replicate agent across devices
    # need the jnp.array to avoid a bug where device_put doesn't recognize primitives
    agent = jax.device_put(jax.tree_map(jnp.array, agent), sharding.replicate())

    timer = Timer()
    for i in tqdm.tqdm(range(int(FLAGS.config.num_steps))):
        timer.tick("total")

        timer.tick("dataset")
        batch = next(train_data_iter)

        if FLAGS.config.dataset_kwargs.goal_relabeling_strategy == "delta_goals_with_generated_encode_decode":
            assert np.max(batch["uses_generated_goal"] + batch["uses_encode_decode_goal"] + batch["uses_noised_encode_decode_goal"]) <= 1, f'batch["uses_generated_goal"]: {batch["uses_generated_goal"]}, batch["uses_encode_decode_goal"]: {batch["uses_encode_decode_goal"]}, batch["uses_noised_encode_decode_goal"]: {batch["uses_noised_encode_decode_goal"]}'
            
            generated_goal_mask = batch["uses_generated_goal"]
            encode_decode_mask = batch["uses_encode_decode_goal"]
            noised_encode_decode_mask = batch["uses_noised_encode_decode_goal"]
            real_goal_mask = np.logical_not(batch["uses_generated_goal"] + batch["uses_encode_decode_goal"] + batch["uses_noised_encode_decode_goal"])
            assert np.array_equal(generated_goal_mask + encode_decode_mask + noised_encode_decode_mask + real_goal_mask, np.ones_like(generated_goal_mask)), f"generated_goal_mask: {generated_goal_mask}, encode_decode_mask: {encode_decode_mask}, encode_decode_mask: {encode_decode_mask}, noised_encode_decode_mask: {noised_encode_decode_mask}"



        # # DEBUG: save the real vs generated goals and stuff, also unaugmented
        # frames = np.concatenate([batch["observations"]["image"], batch["goals"]["image"], batch["goals"]["unaugmented_image"], batch["unaugmented_image_obs"], batch["goals"]["real_image"], batch["goals"]["encode_decode_image"], batch["goals"]["noised_encode_decode_image"], batch["goals"]["generated_image"]], axis=2)

        # new_frames = []
        # for i, frame in enumerate(frames):
        #     # Choose font and scale
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     font_scale = 0.5
        #     font_color = (0, 255, 0)  # White color in BGR
        #     line_type = 2  # Line thickness

        #     # Add text to the image
        #     frame = cv2.putText(frame, f'obs', (10, 15), font, font_scale, font_color, line_type)
        #     frame = cv2.putText(frame, f'selected goal', (210, 15), font, font_scale, font_color, line_type)
        #     frame = cv2.putText(frame, f'unaugmented goal', (410, 15), font, font_scale, font_color, line_type)
        #     frame = cv2.putText(frame, f'unaugmented obs', (610, 15), font, font_scale, font_color, line_type)
        #     frame = cv2.putText(frame, f'real goal', (810, 15), font, font_scale, font_color, line_type)
        #     frame = cv2.putText(frame, f'enc-dec goal', (1010, 15), font, font_scale, font_color, line_type)
        #     frame = cv2.putText(frame, f'noised enc-dec goal', (1210, 15), font, font_scale, font_color, line_type)
        #     frame = cv2.putText(frame, f'generated goal', (1410, 15), font, font_scale, font_color, line_type)

        #     goal_type = "real"
        #     if batch["uses_generated_goal"][i]:
        #         goal_type = "generated"
        #     if batch["uses_encode_decode_goal"][i]:
        #         goal_type = "encode_decode"
        #     if batch["uses_noised_encode_decode_goal"][i]:
        #         goal_type = "noised_encode_decode"

        #     frame = cv2.putText(frame, f'[{i}]: {goal_type}', (10, 185), font, font_scale, font_color, line_type)
        #     new_frames.append(frame)

        # save_video("./visualized_batches/batch.mp4", np.array(new_frames))



        # cv2.puttext on the image for batch["uses_generated_goal"]
        # Figure out the thing where shuffled curr idxs doesn't match curr idxs 
            # probably something to do w trajs having 64 len and the batch size having a different lngth
            # or that the batches aren't in order of the trajectories. Probably this? Idk 
        # DEBUG: Can check that the correct generated goal image is being chosen 
        # for i, uses_generated in enumerate(batch["uses_generated_goal"]):
        #     assert np.array_equal(batch["goals"]["generated_image"][i], batch["generated_goals"][i, batch["idxs_of_generated_goals"][i]])

        #     # print(f'[{i}] uses generated: {uses_generated}: array equal: {np.array_equal(batch["goals"]["generated_image"][i], batch["generated_goals"][i, batch["idxs_of_generated_goals"][i]])}') # This works because it is the original, unaugmented generated images
        #     print(f'[{i}] uses generated: {uses_generated}: array equal: {np.array_equal(batch["goals"]["unaugmented_image"][i], batch["generated_goals"][i, batch["idxs_of_generated_goals"][i]])}, array equal2: {np.array_equal(batch["goals"]["unaugmented_image"][i], batch["goals"]["generated_image"][i])}')
        #     # print(f'[{i}] uses generated: {uses_generated}: array equal: {np.array_equal(batch["goals"]["image"][i], batch["generated_goals"][i, batch["idxs_of_generated_goals"][i]])}')  # wait this doesn't work because the goal image is augmented
        #     # print(f'[{i}] uses generated: {uses_generated}: array equal: {np.array_equal(batch["goals"]["image"][i], batch["goals"]["generated_image"][i])}') # wait this doesn't work because the goal image is augmented

        # DEBUG: print that the correct percentage of generated goals are being used
        # print('batch["uses_generated_goal"].sum() / batch["uses_generated_goal"].shape[0]:', batch["uses_generated_goal"].sum() / batch["uses_generated_goal"].shape[0])


        # if FLAGS.config.dataset_kwargs.use_generated_goals:
        #     batch = add_generated_goals_to_batch(batch, frac_generated=FLAGS.config.frac_generated)


            # x = batch["goals"]["image"].copy()
            # batch = add_generated_goals_to_batch(batch, frac_generated=FLAGS.config.frac_generated)
            # frames = np.concatenate([batch["observations"]["image"][:, 0], x, batch["goals"]["image"]], axis=2)
            # print(np.array_equal(x, batch["goals"]["image"]))
            # save_video("./visualized_batches/batch.mp4", frames)
        
        # print('batch["goal_dists"]:', batch["goal_dists"])
        # print('batch["orig_goal_dists"]:', batch["orig_goal_dists"])
        # print('batch["max_goal_dists"]:', batch["max_goal_dists"])
        # print('batch["uses_generated_goal"]:', batch["uses_generated_goal"])

        batch = shard_batch(batch, sharding) ###$$$###
        
        # batch = shard_batch(next(train_data_iter), sharding)
        timer.tock("dataset")

        timer.tick("train")

        # print("batch.keys():", batch.keys())
        # print("batch[\"observations\"][\"image\"].shape:", batch["observations"]["image"].shape)
        # print("batch[\"actions\"].shape:", batch["actions"].shape)

        # print("batch[\"traj_len\"]:", batch["traj_len"])
        # # print("batch[\"traj_len2\"]:", batch["traj_len2"])
        # print("batch[\"goal_idxs1\"].shape:", batch["goal_idxs1"].shape)
        # print("batch[\"goal_idxs1\"]:", batch["goal_idxs1"])
        # print("batch[\"goal_idxs2\"]:", batch["goal_idxs2"])
        # print("batch[\"goal_reached_mask1\"]:", batch["goal_reached_mask1"])
        # print("batch[\"goal_reached_mask2\"]:", batch["goal_reached_mask2"])

        # print("batch[\"rewards\"].shape:", batch["rewards"].shape)
        # print("batch[\"rewards\"]:", batch["rewards"])

        agent, update_info = agent.update(batch)
        timer.tock("train")

        if (i + 1) % FLAGS.config.eval_interval == 0:
            logging.info("Evaluating...")
            timer.tick("val")
            metrics = []
            # for batch in tqdm.tqdm(val_data.iterator()):
            # for _, batch in zip(tqdm.trange(FLAGS.config.num_val_batches), val_data.iterator()):


            if FLAGS.config.language_conditioned:
                ###$$$###
                # val_data_iter = map(shard_fn, map(process_text, val_data.tf_dataset.as_numpy_iterator()))
                val_data_iter = map(process_text, val_data.tf_dataset.as_numpy_iterator())
            else:
                val_data_iter = val_data.iterator()

            for _, batch in zip(tqdm.trange(FLAGS.config.num_val_batches), val_data_iter):
                rng, val_rng = jax.random.split(rng)
                metrics.append(agent.get_debug_metrics(batch, seed=val_rng))
                # val_results = agent.get_debug_metrics(batch, seed=val_rng)

                # if FLAGS.config.dataset_kwargs.use_generated_goals:
                #     batch = add_generated_goals_to_batch(batch, frac_generated=1.)
                #     val_results_with_generated = agent.get_debug_metrics(batch, seed=val_rng)
                #     val_results.update({key + "_generated_goals":val for key, val in val_results_with_generated.items()})

                # metrics.append(val_results)


            metrics = jax.tree_map(lambda *xs: np.mean(xs), *metrics)
            wandb_logger.log({"validation": metrics}, step=i)
            timer.tock("val")

        if (i + 1) % FLAGS.config.save_interval == 0:
            logging.info("Saving checkpoint...")
            checkpoint_path = checkpoints.save_checkpoint(
                save_dir, agent, step=i + 1, keep=1e6
            )
            logging.info("Saved checkpoint to %s", checkpoint_path)

            if FLAGS.save_to_s3:
                s3_sync_callback.on_train_epoch_end(i + 1)

        timer.tock("total")

        if (i + 1) % FLAGS.config.log_interval == 0:
            update_info = jax.device_get(update_info)
            wandb_logger.log({"training": update_info}, step=i)

            wandb_logger.log({"timer": timer.get_average_times()}, step=i)


def add_generated_goals_to_batch(batch, frac_generated=0.5):

    N = batch["goals"]["image"].shape[0]
    assert batch["generated_goals"].shape[0] == N
    n_generated_goals = batch["generated_goals"].shape[1]

    selected_generated_idxs = np.random.choice(n_generated_goals, size=N)
    generated_goals = batch["generated_goals"][np.arange(N), selected_generated_idxs, ...]

    random_idxs = np.arange(N)
    np.random.shuffle(random_idxs)
    random_idxs = random_idxs[:int(N * frac_generated)]

    batch["goals"]["image"] = batch["goals"]["image"].copy() # Need to do this to overcome the "ValueError: assignment destination is read-only" error
    batch["goals"]["image"][random_idxs] = generated_goals[random_idxs]
    del batch["generated_goals"]
    return batch

if __name__ == "__main__":
    app.run(main)


