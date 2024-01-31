# -------------------------
# Base SageMaker image
# -------------------------
# FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-sagemaker
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker
# See https://github.com/aws/deep-learning-containers/blob/master/available_images.md

ENV PROJECT=bridgev2

# https://gist.github.com/mohanpedala/1e2ff5661761d3abd0385e8223e16425
SHELL ["/bin/bash", "-ex", "-o", "pipefail", "-c"]

# Create an environment setup file separate from /etc/bash.bashrc, and source
# this file from /etc/bash.bashrc. The reason for this is that /etc/bash.bashrc
# doesn't execute when not running the shell interactively. If we have a separate
# file for environment setup, then we can source that file when executing
# non-interactively and still have the expected environment setup.
# This is useful for e.g., running tests with the expected environment.
# Takeaway: Instead of adding stuff to /etc/bash.bashrc, add it to this file!
ENV SHELL_SETUP_FILE=/usr/local/bin/${PROJECT}_env_setup.sh
RUN touch ${SHELL_SETUP_FILE} \
    && chmod +x ${SHELL_SETUP_FILE} \
    && echo "source ${SHELL_SETUP_FILE}" >> /etc/bash.bashrc

# -------------------------
# System tools
# -------------------------
# Configure apt for use in Docker.
RUN echo 'APT::Get::Assume-Yes "true";' | tee /etc/apt/apt.conf.d/90yes
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update -y \
    && apt-get install --no-install-recommends -q -y \
        locales \
        openssh-client openssh-server \
        lsb-release \
        cmake \
        curl \
        docker.io \
        ffmpeg \
        git \
        htop \
        libsm6 \
        libxext6 \
        libgl1  \
        libglib2.0-0 \
        libxrender-dev \
        libxext6 \
        libosmesa6-dev \
        libglfw3 \
        libjpeg-dev \
        libpng-dev \
        patchelf \
        python3-flake8-docstrings \
        python3-pip \
        python3-pyaudio \
        python3-pytest-cov \
        sudo \
        freeglut3-dev \
        software-properties-common \
        tmux \
        libasound2-dev \
    && rm -rf /var/lib/apt/lists/*
RUN locale-gen en_US.UTF-8
ENV \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# # -------------------------
# # Install ROS
# # -------------------------
# ENV ROS_PYTHON_VERSION=3
# # Add ros universe as source and install ros-dev-tools.
# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
#     curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
# RUN add-apt-repository universe
# RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg
# RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
# RUN apt-get update -y \
#     && apt-get install --no-install-recommends -q -y \
#     ros-dev-tools \
#     && rm -rf /var/lib/apt/lists/*
# # Install ros2 python dependencies.
# RUN python3 -m pip install -U \
#     colcon-common-extensions \
#     flake8-blind-except \
#     flake8-builtins \
#     flake8-class-newline \
#     flake8-comprehensions \
#     flake8-deprecated \
#     flake8-import-order \
#     flake8-quotes \
#     "pytest>=5.3" \
#     pytest-repeat \
#     pytest-rerunfailures \
#     catkin_pkg \
#     empy \
#     lark
# # Build ros2 humble from source because humble doesn't distribute ubuntu20.04 binaries.
# RUN mkdir -p /opt/ros/humble/src
# WORKDIR /opt/ros/humble
# RUN vcs import --input https://raw.githubusercontent.com/ros2/ros2/humble/ros2.repos src
# RUN apt-get update -y \
#     && rosdep init \
#     && rosdep update \
#     && rosdep install --from-paths src --ignore-src -y --skip-keys "fastcdr rti-connext-dds-6.0.1 urdfdom_headers" \
#     && rm -rf /var/lib/apt/lists/*
# # Specify the correct python version here otherwise the build uses a mix of versions.
# RUN /opt/conda/bin/python3.9 -m colcon build --symlink-install --packages-skip qt_gui_cpp --packages-skip-by-dep qt_gui_cpp
# RUN echo "source /opt/ros/humble/install/local_setup.bash" >> ${SHELL_SETUP_FILE}

# # Install cv_bridge independently.
# WORKDIR /ros2_ws/src
# RUN git clone https://github.com/ros-perception/vision_opencv.git
# WORKDIR /ros2_ws
# RUN /bin/bash -c "source /opt/ros/humble/install/local_setup.bash; colcon build"
# RUN echo "source /ros2_ws/install/local_setup.bash" >> ${SHELL_SETUP_FILE}

# # -------------------------
# # Install ZED2 SDK
# # -------------------------
# RUN wget https://download.stereolabs.com/zedsdk/4.0/cu118/ubuntu20 -O ZED_SDK_Ubuntu20_cuda11.8_v4.0.2.zstd.run
# RUN chmod +x ./ZED_SDK_Ubuntu20_cuda11.8_v4.0.2.zstd.run && ./ZED_SDK_Ubuntu20_cuda11.8_v4.0.2.zstd.run -- silent
# RUN rm ZED_SDK_Ubuntu20_cuda11.8_v4.0.2.zstd.run
# ENV MESA_GL_VERSION_OVERRIDE 3.3

# -------------------------
# Python packages
# -------------------------
RUN pip install --upgrade pip
# standard deps
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
# Install cython==3.0.0 after other requirements. This conflicts with
# the requirements of robosuite, and breaks some functionality of it,
# but otherwise pip cannot solve for a configuration.
RUN pip install cython==3.0.0
# Install AWS CLI v2 (and uninstall v1). You need v2 for sagemaker sso login.
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install --bin-dir /usr/local/bin --install-dir /usr/local/aws-cli --update && \
    rm awscliv2.zip && \
    pip uninstall -y awscli && \
    aws configure set default.s3.max_concurrent_requests 100 && \
    aws configure set default.s3.max_queue_size 10000

# # ----- put additional custom deps here -----
# RUN pip install open_clip_torch==2.16.0
# RUN export FORCE_CUDA=1 && pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
# RUN pip install kaolin==0.13.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.13.1_cu117.html
# RUN pip install xformers==0.0.16

# # -------------------------
# # Install kaolin-wisp
# # -------------------------
# RUN cd /tmp && git clone https://github.com/NVIDIAGameWorks/kaolin-wisp.git
# RUN cd /tmp/kaolin-wisp && git checkout cb47e10 && export FORCE_CUDA=1 && python setup.py develop

# -------------------------
# Project code
# -------------------------
ENV PATH="/opt/ml/code:${PATH}"

# this environment variable is used by the SageMaker PyTorch container to determine our user code directory.
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code

# /opt/ml and all subdirectories are utilized by SageMaker, use the /code subdirectory to store your user code.
ENV WORKSPACE=/opt/ml/code
COPY . ${WORKSPACE}

WORKDIR ${WORKSPACE}
ENV PYTHONPATH="${WORKSPACE}:$PYTHONPATH"
RUN git config --global --add safe.directory ${WORKSPACE}

# # -------------------------
# # Install ffo
# # -------------------------
# RUN cd /opt/ml/code/external/ffo/vlm/models/grounded_sam/ && pip install -e segment_anything
# RUN cd /tmp && git clone https://github.com/IDEA-Research/GroundingDINO.git
# RUN cd /tmp/GroundingDINO && git checkout 498048b && export FORCE_CUDA=1 && pip install -e .

# # -------------------------
# # Install CoppeliaSim
# # -------------------------
# RUN cd /tmp && \
#     wget https://www.coppeliarobotics.com/files/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz && \
#     tar -xf CoppeliaSim_Edu_V4_1_0_Ubuntu20_04.tar.xz
# ENV COPPELIASIM_ROOT=/tmp/CoppeliaSim_Edu_V4_1_0_Ubuntu20_04
# ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
# ENV QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

# #--------------------------
# # Install PyRep
# # -------------------------
# # We use pip to compile the cffi files in installing PyRep. Note that we do
# # not run `pip install -r requirements.txt`, since cffi is in conflict with
# # the diffusion policy.
# RUN cd /tmp && \
#     git clone https://github.com/stepjam/PyRep.git && \
#     cd PyRep && \
#     pip install --no-deps . 

# Add external packages to python path.
ENV EXTERNAL="${WORKSPACE}/external"
# ENV PYTHONPATH="${EXTERNAL}/diffusion_policy:$PYTHONPATH"
# ENV PYTHONPATH="${EXTERNAL}/datasets:$PYTHONPATH"
# ENV PYTHONPATH="${EXTERNAL}/datasets/externals/camviz:$PYTHONPATH"
# ENV PYTHONPATH="${EXTERNAL}/vidar:$PYTHONPATH"
# ENV PYTHONPATH="${EXTERNAL}/voltron:$PYTHONPATH"
# ENV PYTHONPATH="${EXTERNAL}/ffo:$PYTHONPATH"
# ENV PYTHONPATH="${EXTERNAL}/RLBench:$PYTHONPATH"
ENV PYTHONPATH="${EXTERNAL}/dlimp:$PYTHONPATH"

# # Instead of manually adding the robot_interface to the PYTHONPATH
# # we use pip in order to compile the protobuf files. Note the `--no-deps`
# # command, which avoids overwriting previously-installed dependencies.
# # Because we take this approach, the code referenced when importing
# # this package is not under external/robot_interface, but rather where
# # pip installs packages (we don't use editable mode b/c it doesn't
# # compile the protobufs for some reason). This repo is added as a git
# # submodule as opposed to being installed from github directly in order
# # to avoid storing gredentials in the docker image.
# RUN pip install --no-deps ${EXTERNAL}/robot_interface

# # Compile the ros2 robot policy interface.
# WORKDIR /ros2_ws
# COPY ../external/ros2_robot_policy_interface src/ros2_robot_policy_interface
# RUN /bin/bash -c "source /opt/ros/humble/install/local_setup.bash; colcon build --packages-select ros2_robot_policy_interface"
# WORKDIR WORKSPACE

# RUN cd ./external/dlimp 
# RUN pip install -e .
# RUN cd ../..

# # Need to do these separately for some reason
# RUN pip uninstall -y tensorflow
# RUN pip install tensorflow[and-cuda] 
# RUN pip install tensorflow_datasets>=4.9.2 
# RUN pip install tqdm-multiprocess==0.0.11 
# RUN pip install pre-commit==3.3.3
# RUN pip install typing_extensions

RUN pip install --upgrade "jax[cuda11_pip]"==0.4.11 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    jaxlib==0.4.11 \
    flax==0.7.0 \
    optax==0.1.5 \
    distrax==0.1.2 \
    chex==0.1.7 

RUN pip install orbax-checkpoint==0.4.0


    
# RUN pip install --upgrade typing_extensions
# # RUN pip install ipdb 

# RUN pip uninstall -y jax jaxlib flax optax distrax chex
# RUN pip install --upgrade "jax[cuda11_pip]"==0.4.11 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 
# RUN pip install "jax[cuda]"==0.4.23 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# RUN pip install flax optax distrax chex

RUN alias p3=python3
# Defines default entry point
# TODO: How do we allow for multiple sagemaker entry points?
# I guess we could set that as an argument to docker build.
# Do we even need to set this, or can it be specified in the sagemaker script only?
# ENV SAGEMAKER_PROGRAM="${WORKSPACE}/diffusion_policy/train.py"


# RUN python3 -u download_models.py --config configs/base.py:base