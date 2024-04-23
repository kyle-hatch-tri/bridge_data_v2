# # 2 cores per process
# TPU0="export TPU_VISIBLE_DEVICES=0 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8476 TPU_MESH_CONTROLLER_PORT=8476"
# TPU1="export TPU_VISIBLE_DEVICES=1 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8477 TPU_MESH_CONTROLLER_PORT=8477"
# TPU2="export TPU_VISIBLE_DEVICES=2 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8478 TPU_MESH_CONTROLLER_PORT=8478"
# TPU3="export TPU_VISIBLE_DEVICES=3 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8479 TPU_MESH_CONTROLLER_PORT=8479"

# # 4 cores per process
# TPU01="export TPU_VISIBLE_DEVICES=0,1 TPU_CHIPS_PER_HOST_BOUNDS=1,2,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8476 TPU_MESH_CONTROLLER_PORT=8476"
# TPU23="export TPU_VISIBLE_DEVICES=2,3 TPU_CHIPS_PER_HOST_BOUNDS=1,2,1 TPU_HOST_BOUNDS=1,1,1 TPU_MESH_CONTROLLER_ADDRESS=localhost:8478 TPU_MESH_CONTROLLER_PORT=8478"

# NAME="gcbc_diffusion_policy"

# CMD="python experiments/susie/calvin/calvin_gcbc.py \
#     --config experiments/susie/calvin/configs/gcbc_train_config.py:gc_ddpm_bc \
#     --calvin_dataset_config experiments/susie/calvin/configs/gcbc_data_config.py:all \
#     --name $NAME"

# $CMD


cd "/home/kylehatch/Desktop/hidql/bridge_data_v2"
# cd "/opt/ml/code"

# export CUDA_VISIBLE_DEVICES=0,1

# NAME="gcbc_diffusion_policy"
# CMD="python3 -u calvin_gcbc.py \
#     --config experiments/susie/calvin/configs/gcbc_train_config.py:gc_ddpm_bc \
#     --calvin_dataset_config experiments/susie/calvin/configs/gcbc_data_config.py:all \
#     --name $NAME \
#     --debug=1 \
#     --save_to_s3=0 \
#     --s3_save_uri=s3://kyle-sagemaker-training-outputs \
#     --wandb_proj_name susie_gc_low_level"

# NAME="gcbc_policy"
# CMD="python3 -u calvin_gcbc.py \
#     --config experiments/susie/calvin/configs/gcbc_train_config.py:gc_bc \
#     --calvin_dataset_config experiments/susie/calvin/configs/gcbc_data_config.py:all \
#     --name $NAME \
#     --debug=1 \
#     --log_to_wandb=1 \
#     --seed=34 \
#     --save_to_s3=0 \
#     --s3_save_uri=s3://kyle-sagemaker-training-outputs \
#     --wandb_proj_name susie_gc_low_level"

# NAME="gciql_policy"
# CMD="python3 -u calvin_gcbc.py \
#     --config experiments/susie/calvin/configs/gcbc_train_config.py:gc_iql \
#     --calvin_dataset_config experiments/susie/calvin/configs/gcbc_data_config.py:all \
#     --name $NAME \
#     --debug=1 \
#     --log_to_wandb=0 \
#     --save_to_s3=0 \
#     --s3_save_uri=s3://kyle-sagemaker-training-outputs \
#     --wandb_proj_name susie_gc_low_level"

export WANDB_API_KEY=""

# cd /home/kylehatch/Desktop/hidql/bridge_data_v2
# NAME="gcbc_diffusion_policy_noactnorm"
# CMD="python3 -u calvin_gcbc.py \
#     --config experiments/configs/susie/calvin/configs/gcbc_train_config.py:gc_ddpm_bc_noactnorm \
#     --calvin_dataset_config experiments/configs/susie/calvin/configs/gcbc_data_config.py:all \
#     --name $NAME \
#     --debug=1 \
#     --log_to_wandb=1 \
#     --save_to_s3=1 \
#     --s3_save_uri=s3://kyle-sagemaker-training-outputs \
#     --wandb_proj_name susie_gc_low_level \
#     --seed=77"
# $CMD

# --config experiments/configs/susie/calvin/configs/gcbc_train_config.py:calvin_gcdiffusion_noactnorm-auggoaldiff-sagemaker-b1024 \

cd /home/kylehatch/Desktop/hidql/bridge_data_v2
python3 -u calvin_gcbc.py \
--calvin_dataset_config experiments/configs/susie/calvin/configs/gcbc_data_config.py:all \
--config experiments/configs/susie/calvin/configs/gcbc_train_config.py:calvin_gciql_hparams5-noactnorm-auggoaldiff \
--algo=gcdiffusion \
--description=default \
--debug=1 \
--log_to_wandb=1 \
--save_to_s3=1 \
--s3_save_uri=s3://kyle-sagemaker-training-outputs \
--wandb_proj_name susie_low_level \
--seed=25




cd /home/kylehatch/Desktop/hidql/bridge_data_v2
python3 -u calvin_gcbc.py \
--calvin_dataset_config experiments/configs/susie/calvin/configs/gcbc_data_config.py:libero \
--config experiments/configs/susie/calvin/configs/gcbc_train_config.py:liberosplit1_gciql_hparams5-noactnorm-auggoaldiff \
--algo=gcdiffusion \
--description=default \
--debug=1 \
--log_to_wandb=0 \
--save_to_s3=0 \
--s3_save_uri=s3://kyle-sagemaker-training-outputs \
--wandb_proj_name susie_low_level \
--seed=25

