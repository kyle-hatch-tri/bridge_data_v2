from ml_collections import ConfigDict
from copy import deepcopy

def get_config(config_string):
    dataset, algo, variant = config_string.split("_")
    variant = variant.split("-")

    print(f"dataset: {dataset}, algo: {algo}")

    base_real_config = dict(
        batch_size=256,
        num_val_batches=8,
        num_steps=int(2e6),
        log_interval=1000,
        eval_interval=50_000,
        save_interval=50_000,
        save_dir="/home/kylehatch/Desktop/hidql/bridge_data_v2/results",
        data_path="/home/kylehatch/Desktop/hidql/data/calvin_data_processed/goal_conditioned",
        dataset_name=dataset,
        resume_path=None,
        seed=42,
    )


    if "calvin" in dataset:
        dataset_dir = "calvin_data_processed"
    elif dataset == "liberosplit1":
        dataset_dir = "libero_data_processed_split1"
    elif dataset == "liberosplit2":
        dataset_dir = "libero_data_processed_split2"
    elif dataset == "liberosplit3":
        dataset_dir = "libero_data_processed_split3"
    elif dataset == "liberosplit4":
        dataset_dir = "libero_data_processed_split4"
    else:
        raise ValueError(f"Unsupported dataset: \"{dataset}\".")

    if dataset == "calvin":
        base_real_config["data_path"] = f"/home/kylehatch/Desktop/hidql/data/{dataset_dir}/goal_conditioned"
    elif dataset == "calvinlcbc":
        base_real_config["data_path"] = f"/home/kylehatch/Desktop/hidql/data/{dataset_dir}/language_conditioned"
    else:
        base_real_config["data_path"] = f"/home/kylehatch/Desktop/hidql/data/{dataset_dir}"


    base_data_config = dict(
        shuffle_buffer_size=25_000,
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
        use_float64=False, 
    )

    if "libero" in dataset:
        base_data_config["use_float64"] = True 

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
    




    if algo == "gcdiffusion":
        agent_type = "gc_ddpm_bc"

        dataset_kwargs["obs_horizon"] = 1
        dataset_kwargs["act_pred_horizon"] = 4

        agent_kwargs = dict(
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
    elif algo == "lcdiffusion":
        agent_type = "gc_ddpm_bc"

        
        dataset_kwargs["goal_relabeling_kwargs"] = dict(goal_delta=[0, 20])
        dataset_kwargs["load_language"] = True 
        dataset_kwargs["skip_unlabeled"] = True 
        dataset_kwargs["obs_horizon"] = 1
        dataset_kwargs["act_pred_horizon"] = 4

        agent_kwargs = dict(
            score_network_kwargs=dict(
                time_dim=32,
                num_blocks=3,
                dropout_rate=0.1,
                hidden_dim=256,
                use_layer_norm=True,
            ),
            language_conditioned=True, ###$$$###
            # early_goal_concat=None,
            # shared_goal_encoder=None,
            early_goal_concat=False,
            shared_goal_encoder=False,
            use_proprio=False,
            beta_schedule="cosine",
            diffusion_steps=20,
            action_samples=1,
            repeat_last_step=0,
            learning_rate=3e-4,
            warmup_steps=2000,
            actor_decay_steps=int(2e6),
        )

    elif algo == "contrastivevf":
        agent_type = "stable_contrastive_rl_vf"

        agent_kwargs = dict(
            critic_network_kwargs=dict(
                dropout_rate=0.1,
                # hidden_dims=[256, 256],
                hidden_dims=[1024, 1024],
                use_layer_norm=True,
            ),
            early_goal_concat=True,
            shared_goal_encoder=True,
            use_proprio=False,
            learning_rate=3e-4,
            warmup_steps=2000,
        )
    elif algo == "contrastiverl":
        agent_type = "stable_contrastive_rl"

        agent_kwargs = dict(
            critic_network_kwargs=dict(
                dropout_rate=0.1,
                # hidden_dims=[256, 256],
                hidden_dims=[1024, 1024],
                use_layer_norm=True,
            ),
            early_goal_concat=True,
            shared_goal_encoder=True,
            use_proprio=False,
            learning_rate=3e-4,
            warmup_steps=2000,
        )
    elif algo == "gcdiscriminator":
        
        agent_type = "gc_discriminator"

        agent_kwargs = dict(
            network_kwargs=dict(
                dropout_rate=0.1,
                hidden_dims=[256, 256],
                use_layer_norm=True,
            ),
            early_goal_concat=True,
            shared_goal_encoder=True,
            use_proprio=False,
            learning_rate=3e-4,
            warmup_steps=2000,
        )

        if "zeroobs" in variant:
            agent_kwargs["zero_out_obs"] = True 

    elif algo == "gcbc":
        agent_type = "gc_bc"

        agent_kwargs = dict(
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

    elif algo == "lcbc":
        agent_type = "lc_bc"

        dataset_kwargs["goal_relabeling_kwargs"] = dict(goal_delta=[0, 20])
        dataset_kwargs["load_language"] = True 
        dataset_kwargs["skip_unlabeled"] = True 
        
        agent_kwargs = dict(
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
            early_goal_concat=False,
            shared_goal_encoder=False,
            use_proprio=False,
            learning_rate=3e-4,
            warmup_steps=2000,
        )
    elif algo == "gciql":
        agent_type = "gc_iql"
        
        dataset_kwargs["goal_relabeling_strategy"] = "geometric"
        dataset_kwargs["goal_relabeling_kwargs"] = dict(reached_proportion=0.2, discount=0.25)

        agent_kwargs = dict(
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

        if "hparams2" in variant:
            agent_kwargs["discount"] = 0.99
        elif "hparams2" in variant:
            agent_kwargs["discount"] = 0.99
        elif "hparams3" in variant:
            agent_kwargs["target_update_rate"] = 0.005
        elif "hparams4" in variant:
            agent_kwargs["discount"] = 0.99
            agent_kwargs["expectile"] = 0.7
        elif "hparams5" in variant:
            agent_kwargs["discount"] = 0.99
            agent_kwargs["expectile"] = 0.7
            agent_kwargs["temperature"] = 3


        # discount=0.99,
        # expectile=0.9,
        # temperature=1.0,
        # target_update_rate=0.005,
    else:
        raise ValueError(f"Unsupported algo: \"{algo}\".")


    encoder_kwargs=dict(
                    pooling_method="avg",
                    add_spatial_coordinates=True,
                    act="swish",
                )
    
    

    config = dict(
                agent=agent_type,
                agent_kwargs=agent_kwargs,
                dataset_kwargs=dataset_kwargs,
                encoder="resnetv1-34-bridge", 
                encoder_kwargs=encoder_kwargs,
                language_conditioned=False, 
                **base_real_config,
    )

    if "lc" in algo:
        assert algo[:2] == "lc"
        config["language_conditioned"] = True 
        config["encoder"] = "resnetv1-34-bridge-film"
        config["text_processor"] = "muse_embedding"
        config["text_processor_kwargs"] = dict()
        

    # if "generatedgoals" in variant:
    #     config["dataset_kwargs"]["use_generated_goals"] = True 
    #     # config["frac_generated"] = 0.5 
    #     config["dataset_kwargs"]["goal_relabeling_strategy"] = "delta_goals_with_generated"

    #     goal_relabeling_kwargs = dict(goal_delta=[16, 24])

    #     if "frac0.1" in variant:
    #         config["dataset_kwargs"]["goal_relabeling_kwargs"]["frac_generated"] = 0.1
    #     elif "frac0.25" in variant:
    #         config["dataset_kwargs"]["goal_relabeling_kwargs"]["frac_generated"] = 0.25
    #     elif "frac0.5" in variant:
    #         config["dataset_kwargs"]["goal_relabeling_kwargs"]["frac_generated"] = 0.5
    #     elif "frac0.75" in variant:
    #         config["dataset_kwargs"]["goal_relabeling_kwargs"]["frac_generated"] = 0.75
    #     elif "frac0.9" in variant:
    #         config["dataset_kwargs"]["goal_relabeling_kwargs"]["frac_generated"] = 0.9
    #     else:
    #         raise ValueError(f"Need to specify a valid frac_generated")

    #     config["dataset_kwargs"]["goal_relabeling_kwargs"] = goal_relabeling_kwargs

    if "generatedencdecgoal" in variant:
        config["dataset_kwargs"]["use_generated_goals"] = True 
        config["dataset_kwargs"]["use_encode_decode_goals"] = True 
        config["dataset_kwargs"]["goal_relabeling_strategy"] = "delta_goals_with_generated_encode_decode"

        # goal_relabeling_kwargs = dict(goal_delta=[16, 24])

        config["dataset_kwargs"]["goal_relabeling_kwargs"]["goal_delta"] = [16, 24]
        config["dataset_kwargs"]["goal_relabeling_kwargs"]["zero_goal"] = False

        if "frac0.5" in variant:
            config["dataset_kwargs"]["goal_relabeling_kwargs"]["frac_generated"] = 0.5
            config["dataset_kwargs"]["goal_relabeling_kwargs"]["frac_encode_decode"] = 0.16
            config["dataset_kwargs"]["goal_relabeling_kwargs"]["frac_noised_encode_decode"] = 0.16
        elif "frac0.25" in variant:
            config["dataset_kwargs"]["goal_relabeling_kwargs"]["frac_generated"] = 0.25
            config["dataset_kwargs"]["goal_relabeling_kwargs"]["frac_encode_decode"] = 0.25
            config["dataset_kwargs"]["goal_relabeling_kwargs"]["frac_noised_encode_decode"] = 0.25
        else:
            raise ValueError(f"Need to specify a valid frac_generated")
        
        if "zerogoal" in variant:
            config["dataset_kwargs"]["goal_relabeling_kwargs"]["zero_goal"] = True

    

    if "noactnorm" in variant:
        config["dataset_kwargs"]["normalize_actions"] = False

    if "auggoaldiff" in variant:
        config["dataset_kwargs"]["augment_next_obs_goal_differently"] = True 

    if "sagemaker" in variant:
        config["save_dir"] = "/opt/ml/code/results"
        config["data_path"] = f"/opt/ml/input/data/{dataset_dir}"


    for batch_size in [1024, 2048, 4096, 8192]:
        if f"b{batch_size}" in variant:
            config["batch_size"] = batch_size

    config = ConfigDict(config)
    return config
    # return config, dataset, algo, variant


# def get_config(config_string):


#     base_real_config = dict(
#         batch_size=256,
#         num_val_batches=8,
#         num_steps=int(2e6),
#         log_interval=1000,
#         eval_interval=10_000,
#         save_interval=10_000,
#         save_dir="/home/kylehatch/Desktop/hidql/bridge_data_v2/results",
#         data_path="/home/kylehatch/Desktop/hidql/data/calvin_data_processed/goal_conditioned",
#         dataset_name="calvin",
#         resume_path=None,
#         seed=42,
#     )


    
    

#     base_data_config = dict(
#         shuffle_buffer_size=25000,
#         prefetch_num_batches=20,
#         augment=True,
#         # augment_next_obs_goal_differently=False,
#         augment_next_obs_goal_differently=True,
#         augment_kwargs=dict(
#             random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
#             random_brightness=[0.2],
#             random_contrast=[0.8, 1.2],
#             random_saturation=[0.8, 1.2],
#             random_hue=[0.1],
#             augment_order=[
#                 "random_resized_crop",
#                 "random_brightness",
#                 "random_contrast",
#                 "random_saturation",
#                 "random_hue",
#             ],
#         ),
#         normalize_actions=True, 
#     )

#     # params that need to be specified multiple places
#     normalization_type = "normal"
    
#     dataset_kwargs = dict(
#                     goal_relabeling_strategy="delta_goals",
#                     goal_relabeling_kwargs=dict(goal_delta=[0, 24]),
#                     #goal_relabeling_strategy="uniform",
#                     #goal_relabeling_kwargs=dict(reached_proportion=0.0),
#                     #load_language=True,
#                     #skip_unlabeled=True,
#                     relabel_actions=False,
#                     act_pred_horizon=None,
#                     obs_horizon=None,
#                     **base_data_config,
#                 )
    
#     dataset_ddpm_kwargs = dataset_kwargs.copy()
#     dataset_ddpm_kwargs["obs_horizon"] = 1
#     dataset_ddpm_kwargs["act_pred_horizon"] = 4

#     gc_iql_dataset_kwargs = dataset_kwargs.copy()
#     gc_iql_dataset_kwargs["goal_relabeling_strategy"] = "geometric"
#     gc_iql_dataset_kwargs["goal_relabeling_kwargs"] = dict(reached_proportion=0.2,
#                                                            discount=0.25)
    
#     encoder_kwargs=dict(
#                     pooling_method="avg",
#                     add_spatial_coordinates=True,
#                     act="swish",
#                 )
    
#     gc_ddpm_bc_kwargs = dict(
#             score_network_kwargs=dict(
#                 time_dim=32,
#                 num_blocks=3,
#                 dropout_rate=0.1,
#                 hidden_dim=256,
#                 use_layer_norm=True,
#             ),
#             #language_conditioned=True,
#             early_goal_concat=True,
#             shared_goal_encoder=True,
#             use_proprio=False,
#             beta_schedule="cosine",
#             diffusion_steps=20,
#             action_samples=1,
#             repeat_last_step=0,
#             learning_rate=3e-4,
#             warmup_steps=2000,
#             actor_decay_steps=int(2e6),
#         )
    

#     gc_bc_kwargs = dict(
#             network_kwargs=dict(
#                 dropout_rate=0.1,
#                 hidden_dims=[256, 256],
#                 use_layer_norm=True,
#             ),
#             policy_kwargs=dict(tanh_squash_distribution=False, 
#                                state_dependent_std=False,
#                             #    dropout=0.0,
#                                ),
#             #language_conditioned=True,
#             early_goal_concat=True,
#             shared_goal_encoder=True,
#             use_proprio=False,
#             learning_rate=3e-4,
#             warmup_steps=2000,
#         )
    

#     gc_iql_kwargs = dict(
#             network_kwargs=dict(
#                 dropout_rate=0.1,
#                 hidden_dims=[256, 256],
#                 use_layer_norm=True,
#             ),
#             policy_kwargs=dict(tanh_squash_distribution=False, 
#                                state_dependent_std=False,
#                             #    dropout=0.0,
#                                ),
#             #language_conditioned=True,
#             early_goal_concat=True,
#             shared_goal_encoder=True,
#             use_proprio=False,
#             learning_rate=3e-4,
#             warmup_steps=2000,

#             actor_decay_steps=int(2e6),
#             negative_proportion=0.0,
#             shared_encoder=False,
#             discount=0.95,
#             expectile=0.9,
#             temperature=1.0,
#             target_update_rate=0.002,
#             dropout_target_networks=True,
#         )
    
#     # gc_iql2_kwargs = gc_iql_kwargs.copy()
#     # gc_iql2_kwargs["discount"] = 0.99
#     # gc_iql2_kwargs["target_update_rate"] = 0.005

#     gc_iql2_kwargs = gc_iql_kwargs.copy()
#     gc_iql2_kwargs["discount"] = 0.99

#     gc_iql3_kwargs = gc_iql_kwargs.copy()
#     gc_iql3_kwargs["target_update_rate"] = 0.005

#     gc_iql4_kwargs = gc_iql_kwargs.copy()
#     gc_iql4_kwargs["discount"] = 0.99
#     gc_iql4_kwargs["expectile"] = 0.7

#     gc_iql5_kwargs = gc_iql_kwargs.copy()
#     gc_iql5_kwargs["discount"] = 0.99
#     gc_iql5_kwargs["expectile"] = 0.7
#     gc_iql5_kwargs["temperature"] = 3

#     # discount=0.99,
#     # expectile=0.9,
#     # temperature=1.0,
#     # target_update_rate=0.005,


#     possible_structures = {
#         "calvin_gc_ddpm_bc": ConfigDict(
#             dict(
#                 agent="gc_ddpm_bc",
#                 agent_kwargs=gc_ddpm_bc_kwargs,
#                 dataset_kwargs=dataset_ddpm_kwargs,
#                 encoder="resnetv1-34-bridge",
#                 encoder_kwargs=encoder_kwargs,
#                 **base_real_config,
#             )
#         ),    

#         "gc_bc": ConfigDict(
#             dict(
#                 agent="gc_bc",
#                 agent_kwargs=gc_bc_kwargs,
#                 dataset_kwargs=dataset_kwargs,
#                 encoder="resnetv1-34-bridge",
#                 encoder_kwargs=encoder_kwargs,
#                 **base_real_config,
#             )
#         ),   

#         "gc_iql": ConfigDict(
#             dict(
#                 agent="gc_iql",
#                 agent_kwargs=gc_iql_kwargs,
#                 dataset_kwargs=gc_iql_dataset_kwargs,
#                 encoder="resnetv1-34-bridge",
#                 encoder_kwargs=encoder_kwargs,
#                 **base_real_config,
#             )
#         ),   

#         "gc_iql2": ConfigDict(
#             dict(
#                 agent="gc_iql",
#                 agent_kwargs=gc_iql2_kwargs,
#                 dataset_kwargs=gc_iql_dataset_kwargs,
#                 encoder="resnetv1-34-bridge",
#                 encoder_kwargs=encoder_kwargs,
#                 **base_real_config,
#             )
#         ),   

#         "gc_iql3": ConfigDict(
#             dict(
#                 agent="gc_iql",
#                 agent_kwargs=gc_iql3_kwargs,
#                 dataset_kwargs=gc_iql_dataset_kwargs,
#                 encoder="resnetv1-34-bridge",
#                 encoder_kwargs=encoder_kwargs,
#                 **base_real_config,
#             )
#         ),      

#         "gc_iql4": ConfigDict(
#             dict(
#                 agent="gc_iql",
#                 agent_kwargs=gc_iql4_kwargs,
#                 dataset_kwargs=gc_iql_dataset_kwargs,
#                 encoder="resnetv1-34-bridge",
#                 encoder_kwargs=encoder_kwargs,
#                 **base_real_config,
#             )
#         ),   

#         "gc_iql5": ConfigDict(
#             dict(
#                 agent="gc_iql",
#                 agent_kwargs=gc_iql5_kwargs,
#                 dataset_kwargs=gc_iql_dataset_kwargs,
#                 encoder="resnetv1-34-bridge",
#                 encoder_kwargs=encoder_kwargs,
#                 **base_real_config,
#             )
#         ),   
#     }


#     local_keys = list(possible_structures.keys())
#     for key in local_keys:
#         possible_structures[key + "_noactnorm"] = deepcopy(possible_structures[key])
#         possible_structures[key + "_noactnorm"]["dataset_kwargs"]["normalize_actions"] = False


#     local_keys = list(possible_structures.keys())
#     for batch_size in [1024, 2048, 4096, 8192]:
#         for key in local_keys:
#             possible_structures[key + f"_b{batch_size}"] = deepcopy(possible_structures[key])
#             possible_structures[key + f"_b{batch_size}"]["batch_size"] = batch_size


    
#     # local_keys = list(possible_structures.keys())
#     # for key in local_keys:
#     #     possible_structures[key + "_auggoaldiff"] = deepcopy(possible_structures[key])
#     #     possible_structures[key + "_auggoaldiff"]["dataset_kwargs"]["augment_next_obs_goal_differently"] = True 

#     local_keys = list(possible_structures.keys())
#     for key in local_keys:
#         possible_structures[key + "_sagemaker"] = deepcopy(possible_structures[key])
#         possible_structures[key + "_sagemaker"]["save_dir"] = "/opt/ml/code/results"
#         possible_structures[key + "_sagemaker"]["data_path"] = "/opt/ml/input/data/calvin_data_processed"

#     return possible_structures[config_string]


