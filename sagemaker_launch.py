import argparse
import os
from datetime import datetime

import boto3
import sagemaker
# from sagemaker.pytorch import PyTorch
from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import FileSystemInput


def get_job_name(base):
    now = datetime.now()
    now_ms_str = f'{now.microsecond // 1000:03d}'
    date_str = f"{now.strftime('%Y-%m-%d-%H-%M-%S')}-{now_ms_str}"
    job_name = '_'.join([base, date_str])
    return job_name


def launch(args):

    if args.wandb_api_key is None:
        wandb_api_key = os.environ.get('WANDB_API_KEY', None)
        assert wandb_api_key is not None, 'Please provide wandb api key either via --wandb-api-key or env variable WANDB_API_KEY.'
        args.wandb_api_key = wandb_api_key

    if args.local:
        assert args.instance_count == 1, f'Local mode requires 1 instance, now get {args.instance_count}.'
        assert args.input_source not in {'lustre'}
        args.sagemaker_session = sagemaker.LocalSession()
    else:
        assert args.input_source not in {'local'}    
        args.sagemaker_session = sagemaker.Session()

    if args.input_source == 'local':
        input_mode = 'File'
        # training_inputs = {"calvin_data_processed":'file:///home/kylehatch/Desktop/hidql/data/calvin_data_processed/goal_conditioned'}
        training_inputs = {"calvin_data_processed":'file:///home/kylehatch/Desktop/hidql/data/calvin_data_processed/goal_conditioned',
                           "calvin_data_processed_subset":'file:///home/kylehatch/Desktop/hidql/data/calvin_data_processed_subset/goal_conditioned'}
    elif args.input_source == 'lustre':
        input_mode = 'File'
        train_fs = FileSystemInput(
            file_system_id='fs-02831553b25f26b1c', ###TODO
            file_system_type='FSxLustre',
            directory_path='/onhztbev', ###TODO
            file_system_access_mode='ro'
        )
    elif args.input_source == 's3':
        input_mode = 'FastFile'
        # train_fs = 's3://tri-ml-datasets/scratch/dianchen/datasets/' ###TODO
        training_inputs = {
                        "calvin_data_processed":'s3://susie-data/calvin_data_processed/goal_conditioned/',
                           "calvin_data_processed_subset":'s3://susie-data/calvin_data_processed_subset/goal_conditioned/'}
    else:
        raise ValueError(f'Invalid input source {args.input_source}')

    role = 'arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess'
    role_name = role.split(['/'][-1])

    session = boto3.session.Session()
    region = session.region_name

    config = os.path.join('/opt/ml/code/', args.config)
    calvin_dataset_config = os.path.join('/opt/ml/code/', args.calvin_dataset_config)
    hyperparameters = {
        'config': config,
        'calvin_dataset_config': calvin_dataset_config,
        "name":args.name,
        "debug":args.debug,
        "wandb_proj_name":args.wandb_proj_name,
        "save_to_s3":args.save_to_s3,
        "s3_save_uri":args.s3_save_uri,
        "log_to_wandb":args.log_to_wandb,
        "seed":args.seed,

    }




    subnets = [
        'subnet-05f1115c7d6ccbd07',
        'subnet-03c7c7be28e923670',
        'subnet-0a29e4f1a47443e28',
        'subnet-06e0db77592be2b36',
        'subnet-0dd3f8c4ce7e0ae4c',
        'subnet-02a6ddd2a60a8e048',
        'subnet-060ad40beeb7f24b4',
        'subnet-036abdaead9798455',
        'subnet-07ada213d5ef651bb',
        'subnet-0e260ba29726b9fbb',
        'subnet-08468a58663b2b173',
        'subnet-0ecead4af60b3306f',
        'subnet-09b3b182287e9aa29',
        'subnet-07bf42d7c9cb929e4',
        'subnet-0f72615fd9bd3c717',
        'subnet-0578590f6bd9a5dde',
    ]


    security_group_ids = [
        'sg-0afb9fb0e79a54061', 
        'sg-0333993fea1aeb948', 
        'sg-0c4b828f4023a04cc',
    ]



    job_name = get_job_name(args.base_job_name)

    if args.local:
        image_uri = f'{args.base_job_name}:latest' 
    else:
        image_uri = f'124224456861.dkr.ecr.us-east-1.amazonaws.com/{args.base_job_name}:latest'
    
    output_path = os.path.join(f's3://tri-ml-sandbox-16011-us-east-1-datasets/sagemaker/{args.user}/bridge_data_v2/', job_name)

    checkpoint_s3_uri = None if args.local else output_path
    checkpoint_local_path = None if args.local else '/opt/ml/checkpoints'
    code_location = output_path

    base_job_name = args.base_job_name.replace("_", "-")
    instance_count = args.instance_count
    entry_point = args.entry_point
    sagemaker_session = args.sagemaker_session

    instance_type = 'local_gpu' if args.local else args.instance_type 
    keep_alive_period_in_seconds = 0
    max_run = 60 * 60 * 24 * 5

    environment = {
        # 'PYTHONPATH': '/opt/ml/code:/opt/ml/code/externals/datasets:/opt/ml/code/external/dlimp', 
        'WANDB_API_KEY': args.wandb_api_key,
        'WANDB_ENTITY': "tri",
        # 'EXP_DESCRIPTION': args.exp_description,

        # "CUDA_VISIBLE_DEVICES":"1",
        # "XLA_PYTHON_CLIENT_PREALLOCATE":"false",
    }

    distribution = {
        'smdistributed': {
            'dataparallel': {
                    'enabled': True,
            },
        },
    }

    # inputs = {
    #     'training': train_fs,
    # }

    print()
    print()
    print('#############################################################')
    print(f'SageMaker Execution Role:       {role}')
    print(f'The name of the Execution role: {role_name[-1]}')
    print(f'AWS region:                     {region}')
    print(f'Entry point:                    {entry_point}')
    print(f'Image uri:                      {image_uri}')
    print(f'Job name:                       {job_name}')
    print(f'Configuration file:             {config}')
    print(f'Instance count:                 {instance_count}')
    print(f'Input mode:                     {input_mode}')
    print('#############################################################')
    print()
    print()

    # estimator = PyTorch(
    #     base_job_name=base_job_name,
    #     entry_point=entry_point,
    #     hyperparameters=hyperparameters,
    #     role=role,
    #     image_uri=image_uri,
    #     instance_count=instance_count,
    #     instance_type=instance_type,
    #     environment=environment,
    #     sagemaker_session=sagemaker_session,
    #     subnets=subnets,
    #     security_group_ids=security_group_ids,
    #     keep_alive_period_in_seconds=keep_alive_period_in_seconds,
    #     max_run=max_run,
    #     input_mode=input_mode,
    #     job_name=job_name,
    #     output_path=output_path,
    #     checkpoint_s3_uri=checkpoint_s3_uri,
    #     checkpoint_local_path=checkpoint_local_path,
    #     code_location=code_location,
    #     distribution=distribution,
    # )
    # estimator.fit(inputs=inputs)

    if args.enable_ddp:
        estimator = TensorFlow(
            base_job_name=base_job_name,
            entry_point=entry_point,
            hyperparameters=hyperparameters,
            role=role,
            image_uri=image_uri,
            instance_count=instance_count,
            instance_type=instance_type,
            environment=environment,
            sagemaker_session=sagemaker_session,
            subnets=subnets,
            security_group_ids=security_group_ids,
            keep_alive_period_in_seconds=keep_alive_period_in_seconds,
            max_run=max_run,
            input_mode=input_mode,
            job_name=job_name,
            # output_path=output_path,
            checkpoint_s3_uri=checkpoint_s3_uri,
            checkpoint_local_path=checkpoint_local_path,
            code_location=code_location,
            distribution=distribution,
        )
    else:
        # distribution_fake = {
        #     'smdistributed': {
        #         'dataparallel': {
        #                 'enabled': False,
        #         },
        #     },
        # }

        estimator = TensorFlow(
        base_job_name=base_job_name,
        entry_point=entry_point,
        hyperparameters=hyperparameters,
        role=role,
        image_uri=image_uri,
        instance_count=instance_count,
        instance_type=instance_type,
        environment=environment,
        sagemaker_session=sagemaker_session,
        subnets=subnets,
        security_group_ids=security_group_ids,
        keep_alive_period_in_seconds=keep_alive_period_in_seconds,
        max_run=max_run,
        input_mode=input_mode,
        job_name=job_name,
        # output_path=output_path,
        checkpoint_s3_uri=checkpoint_s3_uri,
        checkpoint_local_path=checkpoint_local_path,
        code_location=code_location,
        # distribution=distribution,
        # distribution=distribution_fake,
    )
    estimator.fit(inputs=training_inputs)

    # estimator = TensorFlow(
    #     role=role,
    #     instance_count=1,
    #     base_job_name="jax",
    #     framework_version="2.10",
    #     py_version="py39",
    #     source_dir="training_scripts",
    #     entry_point="train_jax.py",
    #     instance_type="ml.p3.2xlarge",
    #     hyperparameters={"num_epochs": 3},
    # )
    # estimator.fit(logs=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--calvin_dataset_config', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--debug', type=int, default=0)
    parser.add_argument('--save_to_s3', type=int, default=1)
    parser.add_argument('--log_to_wandb', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--base-job-name', type=str, required=True)
    parser.add_argument('--user', type=str, required=True, help='supported users under the IT-predefined bucket.')
    parser.add_argument('--wandb_proj_name', type=str, default=None)
    parser.add_argument('--s3_save_uri', type=str, default="s3://kyle-sagemaker-training-outputs")
    parser.add_argument('--wandb-api-key', type=str, default=None)
    parser.add_argument('--input-source', choices=['s3', 'lustre', 'local'], default='lustre')
    parser.add_argument('--instance-count', type=int, default=1)
    parser.add_argument('--entry_point', type=str, default='scripts/train.py'),
    parser.add_argument('--enable_ddp', action='store_true', default=False)
    parser.add_argument('--instance_type', type=str, default="ml.p4de.24xlarge"),
    args = parser.parse_args()

    launch(args)

"""
=================== Local debug ===================
./update_docker.sh
python3 -u sagemaker_launch.py \
--entry_point calvin_gcbc.py \
--user kylehatch \
--local \
--input-source local \
--base-job-name bridge_data_v2 \
--debug 1 \
--save_to_s3 0 \
--config experiments/susie/calvin/configs/gcbc_train_config.py:gc_ddpm_bc_sagemaker \
--calvin_dataset_config experiments/susie/calvin/configs/gcbc_data_config.py:all \
--name gcbc_diffusion_policy \
--wandb_proj_name susie_gc_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 


./update_docker.sh
python3 -u sagemaker_launch.py \
--entry_point calvin_gcbc.py \
--user kylehatch \
--local \
--input-source local \
--base-job-name bridge_data_v2 \
--debug 1 \
--log_to_wandb 0 \
--save_to_s3 0 \
--config experiments/susie/calvin/configs/gcbc_train_config.py:gc_bc_sagemaker \
--calvin_dataset_config experiments/susie/calvin/configs/gcbc_data_config.py:all \
--name gcbc_policy \
--wandb_proj_name susie_gc_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--seed 99

=================== Remote debug ===================
./update_docker.sh
./upload_docker.sh
python3 -u sagemaker_launch.py \
--entry_point calvin_gcbc.py \
--user kylehatch \
--input-source s3 \
--base-job-name bridge_data_v2 \
--config experiments/susie/calvin/configs/gcbc_train_config.py:gc_iql_sagemaker \
--calvin_dataset_config experiments/susie/calvin/configs/gcbc_data_config.py:all \
--name gciql \
--debug 1 \
--save_to_s3 0 \
--wandb_proj_name susie_gc_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--instance_type ml.p4d.24xlarge


=================== Remote no debug ===================
./update_docker.sh
./upload_docker.sh
python3 -u sagemaker_launch.py \
--entry_point calvin_gcbc.py \
--user kylehatch \
--input-source s3 \
--base-job-name bridge_data_v2 \
--config experiments/susie/calvin/configs/gcbc_train_config.py:gc_ddpm_bc_sagemaker \
--calvin_dataset_config experiments/susie/calvin/configs/gcbc_data_config.py:all \
--name gcbc_diffusion \
--wandb_proj_name susie_gc_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--instance_type ml.p4d.24xlarge



./update_docker.sh
./upload_docker.sh
python3 -u sagemaker_launch.py \
--entry_point calvin_gcbc.py \
--user kylehatch \
--input-source s3 \
--base-job-name bridge_data_v2 \
--config experiments/susie/calvin/configs/gcbc_train_config.py:gc_ddpm_bc_noactnorm_sagemaker \
--calvin_dataset_config experiments/susie/calvin/configs/gcbc_data_config.py:all \
--name gcbc_diffusion_noactnorm \
--wandb_proj_name susie_gc_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--instance_type ml.p4d.24xlarge


### HEre ###
export AWS_ACCESS_KEY_ID="ASIARZ3C2ZCO5B5IKJ6U"
export AWS_SECRET_ACCESS_KEY="4/tY6CQYt5Putd0+5D7hNu+r2s53SaR+RvE6XeCG"
export AWS_SESSION_TOKEN="IQoJb3JpZ2luX2VjEL3//////////wEaCXVzLWVhc3QtMSJGMEQCIHdQC7SVK3/bGeBb07TDdx95btb9Cyc5Od+MjW973mg3AiA/1zSKNyTgwRTZU/UkGaigUwZVXMjlD8Mb/sC5WOGZaiqSAwiW//////////8BEAQaDDEyNDIyNDQ1Njg2MSIM8NpBfhVv9DZqDBG7KuYCBiDcvX9fWVwHMCnN2U4YBpDKLNrP7FpseXoCz6+XFnt3BYQpKwV0Jrs4+rujVqumbYdKZu5d/CvG2QlC+Zij0TjWjnKWsskRMInH5b+PjZ8t4P5tYlkj0S0QFjXT8jftXmJ/ZXQpz7WPQYPP2efCS/jGxSOasU0/eUGoBTFP9JV5mbbqxtwCrPEWiBgh9UJnYY/oIyumBiQmtMEFmPUOHI/P2W5QcanGfrwe2qWsig6I2OthxZfbg8SqZGT0+mwMAGozGrvAY+jhaIsxL8F9lwCqnrfYUXSnZBS/cEumPHoyN5y63tsgF+GcF2jg46o9LefxvUZFo7yX655QT2lTKg9SO7/GkYuoOZYYdti3/VChJ4bNshIwenGQaKafOamk8yyi2pgbb21HuImptww1oHj+f5XiIBXPpd8u6iOh18IKU7RHlHZP5A3+0kocZCXjCg4YNYhc5bzJxPSSZ/SVx6yq8/cSPTCL1bSuBjqnAd69msZVjJy4b6RRTqcYKEEweN2VX7kbtcehccfuhW60VOcoA3w8qEr4iqil+5vJMFFHbOonjbYilnXdXhGbAxH4ZSJKJyIJHQLUgCKgCbxuuXwaOWLvNSR9lo5EgignVDzhU3jhHIt5iV/SrzvHnVioMY3JzPP85ATHjaiM/7hUjMfZ3zpVVNu3C1WNB283JgEPZOSdzn1v6/Jmnf9n6Aqc8pRAv99f"
./update_docker.sh
./upload_docker.sh
python3 -u sagemaker_launch.py \
--entry_point calvin_gcbc.py \
--user kylehatch \
--input-source s3 \
--base-job-name bridge_data_v2 \
--config experiments/susie/calvin/configs/gcbc_train_config.py:gc_bc_noactnorm_sagemaker \
--calvin_dataset_config experiments/susie/calvin/configs/gcbc_data_config.py:all \
--name gcbcnam \
--wandb_proj_name susie_gc_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--instance_type ml.p4d.24xlarge \
--seed 0

export AWS_ACCESS_KEY_ID="ASIARZ3C2ZCOX3OGOVW7"
export AWS_SECRET_ACCESS_KEY="j7S4bL0ynzANExZKR1o20nNoPS45J90501b/7FtM"
export AWS_SESSION_TOKEN="IQoJb3JpZ2luX2VjELz//////////wEaCXVzLWVhc3QtMSJHMEUCIEupxOSdnKad9V9bmIyxaYZRVJ8ES7XHUXk1VYLtgdJ9AiEAsCXmua9g6cxyNtnTM65P05jr1X45U9nktPwVFyXWlewqkgMIlf//////////ARAEGgwxMjQyMjQ0NTY4NjEiDJi0gdb8AZKcYpT/jCrmAl9O/LICdF/Aqs1PQ8F+6MWtX2KdGuiKKeAVc1u81mu1FxTYFh+HNncBZ5qYJrO3Xd4ffSW+8FQDn2hAt5NMXGAda01wJWyRLlGl28lMpEPMMbJgN4jxPZd8ccZ869Ac57ml1TRA5oQnwKUFcE/YqXNa2I773xz8gDgwOGsBxuTHkvsZG+1qpOYSoUCeP2ipkF5sIi9XaGNDNhMppo3xR8N3Q6p8bEsJykO0uRtUpRhlKnQDP29Pu8Fxfg6oCX1x9h2NtQAgn/IDs58E/+9NY8rFPB7HGlCiww4tZDOLAMD7CM/c4CQLRioNU1TBFBsKddWSwwguXPkVLM6XELcHh6ysveyRP2k8djVcvMMzDwngNAzDYSOwcOfZAre4GJCeSi4cDpy3iAc/n+xuPO6Y2HvyEuwxkfMDN25fEIDcU450fqHuUAFwF8xGNDc0ePPopV4OR/pfe2lzKekqcQC1ybe7jsZCz7ww7q60rgY6pgHsOzTE3N9fblkQiOf0+qNZegHCqm7glNFgDAGMV/ssxc+cZCGCOaMfxVs/pzy7ZS+rwg3pm02rY8I/Q4LHMRqVk9FppcpojySO69xOUIhcGofBL4GHQ7+L/IU1UjVSJkWjfYc6P+SmusDkPDaHJdf6e7eO1DN8zGB/zTB3nBZ+4cx1qowVTSmQ42pK854dTzyqf1H3b3soVYv/BOXdyQEcLl6tdTx6"
./update_docker.sh
./upload_docker.sh
python3 -u sagemaker_launch.py \
--entry_point calvin_gcbc.py \
--user kylehatch \
--input-source s3 \
--base-job-name bridge_data_v2 \
--config experiments/susie/calvin/configs/gcbc_train_config.py:gc_iql5_noactnorm_sagemaker \
--calvin_dataset_config experiments/susie/calvin/configs/gcbc_data_config.py:all \
--name gciql5nam \
--wandb_proj_name susie_gc_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--instance_type ml.p4d.24xlarge \
--seed 0







###
python3 -u sagemaker_launch.py \
--entry_point calvin_gcbc.py \
--user kylehatch \
--input-source s3 \
--base-job-name bridge_data_v2 \
--config experiments/susie/calvin/configs/gcbc_train_config.py:gc_bc_sagemaker \
--calvin_dataset_config experiments/susie/calvin/configs/gcbc_data_config.py:all \
--name gcbc \
--wandb_proj_name susie_gc_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--seed 0 \
--local \
--debug 1 \
--save_to_s3 0 


export AWS_ACCESS_KEY_ID="ASIARZ3C2ZCOSPGKYREF"
export AWS_SECRET_ACCESS_KEY="VgOjnCDrhni92DXKkPfDe1+30l6ER3Abpp+45A2Y"
export AWS_SESSION_TOKEN="IQoJb3JpZ2luX2VjEDEaCXVzLWVhc3QtMSJHMEUCIEnDoRd6B4D/NE0Si5NbDV/wHNIcnooqYFnTkVdPQD/qAiEAzhnLggtVdWmFrlqbunwEqtQIwY4Ix/OiSJeC9KsGK/UqkgMI+f//////////ARAEGgwxMjQyMjQ0NTY4NjEiDHVaVEdBWLPR2K98syrmAmmtEMUTYnyMtZs1iajhhJ+YlOtAotxY6o3Z3Qs+2+FlmB+mdVMACPQ6u7uTwRHMd8fm6geYZVxfgKo/ra2OIlI7IsCYGJ8P9oWVzgARdA7VdhGG0i9QKGyd8ml3sghCHqQRi+QBJ70lCYxe7zRDRmKdjAE47EyxtrVuJGS+R9qDsSSfZ3dJt5b6orCDTnPj3Z7Rnoi0vZX7/EFCrqXNnIG1wCpLRpaKL5pYOkxe7xqApvH3hQ6u1A6GHgtT1GaO4/FMyEGPWKEeC0v4ueBevPgXdQkARD9ZFUmRievHxByKhXKynfl1EjKht8poW46RF0f2xW+JH1nRqWcKD7b1uhemdIzGvTydVtM3m+IgmGYPMWiqplX9Ld74KO3GYjkbdkTg/a1+3CxvNt62DcWKJ1owh3ZVq/QQAW3f9evSuNdxK4RHhr/aKGRXBOoZ2S/Anb0wVAwfDl0/xhRvZwbAol5GKUm3DJwwg92VrgY6pgGaoMdZbiK9K/EBiPnFvORmSwGvRqI8ij5iosq+13cG7HDgxXntmD2VyPJELgT/3waOYNbMRjhUujjOMsouOqWsvzkM80/6FFWW7kEpYK7jYpMBazPliLTtkNudGa7gD74SUkhKv3zojNpVbiag8+XYzqAziXEaMnIYU37IO3ZJVr/v1VM4mlsTpDtksgJ7laXRlDgPC7daQawSDyQHldK0ojDQGGRR"
python3 -u sagemaker_launch.py \
--entry_point calvin_gcbc.py \
--user kylehatch \
--input-source s3 \
--base-job-name bridge_data_v2 \
--config experiments/susie/calvin/configs/gcbc_train_config.py:gc_bc_sagemaker \
--calvin_dataset_config experiments/susie/calvin/configs/gcbc_data_config.py:all \
--name gcbc \
--wandb_proj_name susie_gc_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--instance_type ml.p4d.24xlarge \
--seed 2

export AWS_ACCESS_KEY_ID="ASIARZ3C2ZCOSPGKYREF"
export AWS_SECRET_ACCESS_KEY="VgOjnCDrhni92DXKkPfDe1+30l6ER3Abpp+45A2Y"
export AWS_SESSION_TOKEN="IQoJb3JpZ2luX2VjEDEaCXVzLWVhc3QtMSJHMEUCIEnDoRd6B4D/NE0Si5NbDV/wHNIcnooqYFnTkVdPQD/qAiEAzhnLggtVdWmFrlqbunwEqtQIwY4Ix/OiSJeC9KsGK/UqkgMI+f//////////ARAEGgwxMjQyMjQ0NTY4NjEiDHVaVEdBWLPR2K98syrmAmmtEMUTYnyMtZs1iajhhJ+YlOtAotxY6o3Z3Qs+2+FlmB+mdVMACPQ6u7uTwRHMd8fm6geYZVxfgKo/ra2OIlI7IsCYGJ8P9oWVzgARdA7VdhGG0i9QKGyd8ml3sghCHqQRi+QBJ70lCYxe7zRDRmKdjAE47EyxtrVuJGS+R9qDsSSfZ3dJt5b6orCDTnPj3Z7Rnoi0vZX7/EFCrqXNnIG1wCpLRpaKL5pYOkxe7xqApvH3hQ6u1A6GHgtT1GaO4/FMyEGPWKEeC0v4ueBevPgXdQkARD9ZFUmRievHxByKhXKynfl1EjKht8poW46RF0f2xW+JH1nRqWcKD7b1uhemdIzGvTydVtM3m+IgmGYPMWiqplX9Ld74KO3GYjkbdkTg/a1+3CxvNt62DcWKJ1owh3ZVq/QQAW3f9evSuNdxK4RHhr/aKGRXBOoZ2S/Anb0wVAwfDl0/xhRvZwbAol5GKUm3DJwwg92VrgY6pgGaoMdZbiK9K/EBiPnFvORmSwGvRqI8ij5iosq+13cG7HDgxXntmD2VyPJELgT/3waOYNbMRjhUujjOMsouOqWsvzkM80/6FFWW7kEpYK7jYpMBazPliLTtkNudGa7gD74SUkhKv3zojNpVbiag8+XYzqAziXEaMnIYU37IO3ZJVr/v1VM4mlsTpDtksgJ7laXRlDgPC7daQawSDyQHldK0ojDQGGRR"
python3 -u sagemaker_launch.py \
--entry_point calvin_gcbc.py \
--user kylehatch \
--input-source s3 \
--base-job-name bridge_data_v2 \
--config experiments/susie/calvin/configs/gcbc_train_config.py:gc_iql5_sagemaker \
--calvin_dataset_config experiments/susie/calvin/configs/gcbc_data_config.py:all \
--name gciql5 \
--wandb_proj_name susie_gc_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--instance_type ml.p4d.24xlarge \
--seed 2







python3 -u sagemaker_launch.py \
--entry_point calvin_gcbc.py \
--user kylehatch \
--input-source s3 \
--base-job-name bridge_data_v2 \
--config experiments/susie/calvin/configs/gcbc_train_config.py:gc_iql_sagemaker \
--calvin_dataset_config experiments/susie/calvin/configs/gcbc_data_config.py:all \
--name gciql \
--wandb_proj_name susie_gc_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--instance_type ml.p4d.24xlarge \
--seed 42

python3 -u sagemaker_launch.py \
--entry_point calvin_gcbc.py \
--user kylehatch \
--input-source s3 \
--base-job-name bridge_data_v2 \
--config experiments/susie/calvin/configs/gcbc_train_config.py:gc_iql2_sagemaker \
--calvin_dataset_config experiments/susie/calvin/configs/gcbc_data_config.py:all \
--name gciql2 \
--wandb_proj_name susie_gc_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--instance_type ml.p4d.24xlarge \
--seed 42 

python3 -u sagemaker_launch.py \
--entry_point calvin_gcbc.py \
--user kylehatch \
--input-source s3 \
--base-job-name bridge_data_v2 \
--config experiments/susie/calvin/configs/gcbc_train_config.py:gc_iql3_sagemaker \
--calvin_dataset_config experiments/susie/calvin/configs/gcbc_data_config.py:all \
--name gciql3 \
--wandb_proj_name susie_gc_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--instance_type ml.p4d.24xlarge \
--seed 42 

python3 -u sagemaker_launch.py \
--entry_point calvin_gcbc.py \
--user kylehatch \
--input-source s3 \
--base-job-name bridge_data_v2 \
--config experiments/susie/calvin/configs/gcbc_train_config.py:gc_iql4_sagemaker \
--calvin_dataset_config experiments/susie/calvin/configs/gcbc_data_config.py:all \
--name gciql4 \
--wandb_proj_name susie_gc_low_level \
--wandb-api-key 65915e3ae3752bc3ddc4b7eef1b066067b9d1cb1 \
--instance_type ml.p4d.24xlarge \
--seed 42 




Wonder how much better the official diffusion policy would do than this jax version? 
    > Or robomimic version? 


['/opt/ml/input/data/calvin_data_processed/training/A/traj15/0.tfrecord', '/opt/ml/input/data/calvin_data_processed/training/A/traj15/1.tfrecord', '/opt/ml/input/data/calvin_data_processed/train
ing/A/traj3/0.tfrecord', '/opt/ml/input/data/calvin_data_processed/training/A/traj3/1.tfrecord', '/opt/ml/input/data/calvin_data_processed/training/A/traj16/0.tfrecord', '/opt/ml/input/data/calv
in_data_processed/training/A/traj10/0.tfrecord', '/opt/ml/input/data/calvin_data_processed/training/A/traj10/1.tfrecord', '/opt/ml/input/data/calvin_data_processed/training/A/traj10/2.tfrecord',
 '/opt/ml/input/data/calvin_data_processed/training/A/traj10/3.tfrecord', '/opt/ml/input/data/calvin_data_processed/training/A/traj2/0.tfrecord', '/opt/ml/input/data/calvin_data_processed/traini
ng/A/traj2/1.tfrecord', '/opt/ml/input/data/calvin_data_processed/training/A/traj2/2.tfrecord', '/opt/ml/input/data/calvin_data_processed/training/A/traj2/3.tfrecord', '/opt/ml/input/data/calvin
_data_processed/training/A/traj2/4.tfrecord', '/opt/ml/input/data/calvin_data_processed/training/A/traj2/5.tfrecord', '/opt/ml/input/data/calvin_data_processed/training/A/traj2/6.tfrecord', '/op
t/ml/input/data/calvin_data_processed/training/A/traj1/0.tfrecord', '/opt/ml/input/data/calvin_data_processed/training/A/traj1/1.tfrecord', '/opt/ml/input/data/calvin_data_processed/training/A/t
raj30/0.tfrecord', '/opt/ml/input/data/calvin_data_processed/training/A/traj30/1.tfrecord', '/opt/ml/input/data/calvin_data_processed/training/A/traj30/2.tfrecord', '/opt/ml/input/data/calvin_da
ta_processed/training/A/traj30/3.tfrecord', '/opt/ml/input/data/calvin_data_processed/training/A/traj14/0.tfrecord', '/opt/ml/input/data/calvin_data_processed/training/A/traj14/1.tfrecord', '/op
t/ml/input/data/calvin_data_processed/training/A/traj14/10.tfrecord']                                                                                                                             
len(train_paths[0]):                                                                                                                                                                              
1872        

Installing collected packages: pytz, urllib3, tzdata, tblib, sniffio, smdebug-rulesconfig, schema, rpds-py, pydantic, ppft, pox, platformdirs, jmespath, importlib-metadata, h11, dill, cloudpickle, attrs, uvicorn, referencing, pandas, multiprocess, botocore, anyio, starlette, s3transfer, pathos, jsonschema-specifications, docker, jsonschema, fastapi, boto3, sagemaker
  Attempting uninstall: urllib3
    Found existing installation: urllib3 2.0.7
    Uninstalling urllib3-2.0.7:
      Successfully uninstalled urllib3-2.0.7
  Attempting uninstall: cloudpickle
    Found existing installation: cloudpickle 3.0.0
    Uninstalling cloudpickle-3.0.0:
      Successfully uninstalled cloudpickle-3.0.0
Successfully installed anyio-4.2.0 attrs-23.2.0 boto3-1.34.23 botocore-1.34.23 cloudpickle-2.2.1 dill-0.3.7 docker-7.0.0 fastapi-0.95.2 h11-0.14.0 importlib-metadata-6.11.0 jmespath-1.0.1 jsonschema-4.21.1 jsonschema-specifications-2023.12.1 multiprocess-0.70.15 pandas-2.1.4 pathos-0.3.1 platformdirs-4.1.0 pox-0.3.3 ppft-1.7.6.7 pydantic-1.10.14 pytz-2023.3.post1 referencing-0.32.1 rpds-py-0.17.1 s3transfer-0.10.0 sagemaker-2.203.1 schema-0.7.5 smdebug-rulesconfig-1.0.1 sniffio-1.3.0 starlette-0.27.0 tblib-2.0.0 tzdata-2023.4 urllib3-1.26.18 uvicorn-0.22.0
"""