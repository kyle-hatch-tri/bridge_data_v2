from ml_collections import ConfigDict
from copy import deepcopy

def get_config(config_string):
    base_real_config = dict(
        batch_size=256,
        num_val_batches=8,
        num_steps=int(2e6),
        log_interval=1000,
        eval_interval=10_000,
        save_interval=10_000,
        save_dir="/home/kylehatch/Desktop/hidql/bridge_data_v2/results",
        data_path="/home/kylehatch/Desktop/hidql/data/calvin_data_processed/goal_conditioned",
        resume_path=None,
        seed=42,
    )
    

    base_data_config = dict(
        shuffle_buffer_size=25000,
        prefetch_num_batches=20,
        augment=True,
        augment_next_obs_goal_differently=False,
        augment_kwargs=dict(
            random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
            random_brightness=[0.2],
            random_contrast=[0.8, 1.2],
            random_saturation=[0.8, 1.2],
            random_hue=[0.1],
            augment_order=[
                "random_resized_crop",
                "random_brightness",
                "random_contrast",
                "random_saturation",
                "random_hue",
            ],
        ),
        normalize_actions=True, 
    )

    # params that need to be specified multiple places
    normalization_type = "normal"
    
    dataset_kwargs = dict(
                    goal_relabeling_strategy="delta_goals",
                    goal_relabeling_kwargs=dict(goal_delta=[0, 24]),
                    #goal_relabeling_strategy="uniform",
                    #goal_relabeling_kwargs=dict(reached_proportion=0.0),
                    #load_language=True,
                    #skip_unlabeled=True,
                    relabel_actions=False,
                    act_pred_horizon=None,
                    obs_horizon=None,
                    **base_data_config,
                )
    
    dataset_ddpm_kwargs = dataset_kwargs.copy()
    dataset_ddpm_kwargs["obs_horizon"] = 1
    dataset_ddpm_kwargs["act_pred_horizon"] = 4

    gc_iql_dataset_kwargs = dataset_kwargs.copy()
    gc_iql_dataset_kwargs["goal_relabeling_strategy"] = "geometric"
    gc_iql_dataset_kwargs["goal_relabeling_kwargs"] = dict(reached_proportion=0.2,
                                                           discount=0.25)
    
    encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                )
    
    gc_ddpm_bc_kwargs = dict(
            score_network_kwargs=dict(
                time_dim=32,
                num_blocks=3,
                dropout_rate=0.1,
                hidden_dim=256,
                use_layer_norm=True,
            ),
            #language_conditioned=True,
            early_goal_concat=True,
            shared_goal_encoder=True,
            use_proprio=False,
            beta_schedule="cosine",
            diffusion_steps=20,
            action_samples=1,
            repeat_last_step=0,
            learning_rate=3e-4,
            warmup_steps=2000,
            actor_decay_steps=int(2e6),
        )
    

    gc_bc_kwargs = dict(
            network_kwargs=dict(
                dropout_rate=0.1,
                hidden_dims=[256, 256],
                use_layer_norm=True,
            ),
            policy_kwargs=dict(tanh_squash_distribution=False, 
                               state_dependent_std=False,
                            #    dropout=0.0,
                               ),
            #language_conditioned=True,
            early_goal_concat=True,
            shared_goal_encoder=True,
            use_proprio=False,
            learning_rate=3e-4,
            warmup_steps=2000,
        )
    

    gc_iql_kwargs = dict(
            network_kwargs=dict(
                dropout_rate=0.1,
                hidden_dims=[256, 256],
                use_layer_norm=True,
            ),
            policy_kwargs=dict(tanh_squash_distribution=False, 
                               state_dependent_std=False,
                            #    dropout=0.0,
                               ),
            #language_conditioned=True,
            early_goal_concat=True,
            shared_goal_encoder=True,
            use_proprio=False,
            learning_rate=3e-4,
            warmup_steps=2000,

            actor_decay_steps=int(2e6),
            negative_proportion=0.0,
            shared_encoder=False,
            discount=0.95,
            expectile=0.9,
            temperature=1.0,
            target_update_rate=0.002,
            dropout_target_networks=True,
        )
    
    # gc_iql2_kwargs = gc_iql_kwargs.copy()
    # gc_iql2_kwargs["discount"] = 0.99
    # gc_iql2_kwargs["target_update_rate"] = 0.005

    gc_iql2_kwargs = gc_iql_kwargs.copy()
    gc_iql2_kwargs["discount"] = 0.99

    gc_iql3_kwargs = gc_iql_kwargs.copy()
    gc_iql3_kwargs["target_update_rate"] = 0.005

    gc_iql4_kwargs = gc_iql_kwargs.copy()
    gc_iql4_kwargs["discount"] = 0.99
    gc_iql4_kwargs["expectile"] = 0.7

    gc_iql5_kwargs = gc_iql_kwargs.copy()
    gc_iql5_kwargs["discount"] = 0.99
    gc_iql5_kwargs["expectile"] = 0.7
    gc_iql5_kwargs["temperature"] = 3

    # discount=0.99,
    # expectile=0.9,
    # temperature=1.0,
    # target_update_rate=0.005,


    possible_structures = {
        "gc_ddpm_bc": ConfigDict(
            dict(
                agent="gc_ddpm_bc",
                agent_kwargs=gc_ddpm_bc_kwargs,
                dataset_kwargs=dataset_ddpm_kwargs,
                encoder="resnetv1-34-bridge",
                encoder_kwargs=encoder_kwargs,
                **base_real_config,
            )
        ),    

        "gc_bc": ConfigDict(
            dict(
                agent="gc_bc",
                agent_kwargs=gc_bc_kwargs,
                dataset_kwargs=dataset_kwargs,
                encoder="resnetv1-34-bridge",
                encoder_kwargs=encoder_kwargs,
                **base_real_config,
            )
        ),   

        "gc_iql": ConfigDict(
            dict(
                agent="gc_iql",
                agent_kwargs=gc_iql_kwargs,
                dataset_kwargs=gc_iql_dataset_kwargs,
                encoder="resnetv1-34-bridge",
                encoder_kwargs=encoder_kwargs,
                **base_real_config,
            )
        ),   

        "gc_iql2": ConfigDict(
            dict(
                agent="gc_iql",
                agent_kwargs=gc_iql2_kwargs,
                dataset_kwargs=gc_iql_dataset_kwargs,
                encoder="resnetv1-34-bridge",
                encoder_kwargs=encoder_kwargs,
                **base_real_config,
            )
        ),   

        "gc_iql3": ConfigDict(
            dict(
                agent="gc_iql",
                agent_kwargs=gc_iql3_kwargs,
                dataset_kwargs=gc_iql_dataset_kwargs,
                encoder="resnetv1-34-bridge",
                encoder_kwargs=encoder_kwargs,
                **base_real_config,
            )
        ),      

        "gc_iql4": ConfigDict(
            dict(
                agent="gc_iql",
                agent_kwargs=gc_iql4_kwargs,
                dataset_kwargs=gc_iql_dataset_kwargs,
                encoder="resnetv1-34-bridge",
                encoder_kwargs=encoder_kwargs,
                **base_real_config,
            )
        ),   

        "gc_iql5": ConfigDict(
            dict(
                agent="gc_iql",
                agent_kwargs=gc_iql5_kwargs,
                dataset_kwargs=gc_iql_dataset_kwargs,
                encoder="resnetv1-34-bridge",
                encoder_kwargs=encoder_kwargs,
                **base_real_config,
            )
        ),   
    }


    local_keys = list(possible_structures.keys())
    for key in local_keys:
        possible_structures[key + "_noactnorm"] = deepcopy(possible_structures[key])
        possible_structures[key + "_noactnorm"]["dataset_kwargs"]["normalize_actions"] = False


    local_keys = list(possible_structures.keys())
    for batch_size in [1024, 2048, 4096, 8192]:
        for key in local_keys:
            possible_structures[key + f"_b{batch_size}"] = deepcopy(possible_structures[key])
            possible_structures[key + f"_b{batch_size}"]["batch_size"] = batch_size
    

    local_keys = list(possible_structures.keys())
    for key in local_keys:
        possible_structures[key + "_sagemaker"] = deepcopy(possible_structures[key])
        possible_structures[key + "_sagemaker"]["save_dir"] = "/opt/ml/code/results"
        possible_structures[key + "_sagemaker"]["data_path"] = "/opt/ml/input/data/calvin_data_processed"

    return possible_structures[config_string]








