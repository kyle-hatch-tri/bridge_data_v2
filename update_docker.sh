
# Make the docker image
algorithm_name=bridge_data_v2

docker build -f update.dockerfile \
--build-arg BASE_DOCKER=${algorithm_name} \
-t ${algorithm_name} .
# -t ${algorithm_name}_update .

# # upload the docker image 

# account=$(aws sts get-caller-identity --query Account --output text)

# # Ensure this is 124224456861; if not, see FAQ below.
# echo "account: $account"



# # Currently only us-east-1 is available to us
# region=us-east-1

# # Full name for the docker image
# fullname="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:latest"
# echo "fullname: $fullname"

# aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${fullname}

# docker tag ${algorithm_name}_update ${fullname}
# docker push ${fullname}