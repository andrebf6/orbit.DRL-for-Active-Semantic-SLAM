# Extension Template for Orbit

## Overview

Software developed to train a deep reinforcement learning (DRL) agent for exploration tasks in indoor environments, exploiting semantic information from a 3D scene graph.

This repository is build on topof the [Extension Template for Orbit](https://github.com/isaac-orbit/orbit.ext_template).

## Setup

### Dependencies
- Isaac Sim
- Orbit (SimLab)

### Enviornment Setup
This project has been developed inside a Docker container on a remote server. The GUI can be visualized through the NVIDIA Streaming Client.

1. **Setup a Docker Container with ISAAC SIM**

Create the conteiner:
```js
docker run -d --name isaac-sim --entrypoint bash -it --gpus all -e "ACCEPT_EULA=Y" --rm --network=host     -e "PRIVACY_CONSENT=Y"   -v /home/s3098982/isaac_sim_data:/isaac-sim/data  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw     -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw     -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw     -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw     -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw     -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw     -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw     -v ~/docker/isaac-sim/documents:/root/Documents:rw     nvcr.io/nvidia/isaac-sim:2023.1.1
```
Access its bash shell:
```js
docker exec -it isaac-sim bash
```

Add general features:
```js
cd ..
apt-get update
apt install git-all
```

2. **Install orbit inside the container**

```js
git clone https://github.com/NVIDIA-Omniverse/orbit.git
export ISAACSIM_PATH="/isaac-sim"
export ISAACSIM_PYTHON_EXE="/isaac-sim/python.sh"

cd orbit
ln -s ${ISAACSIM_PATH} _isaac_sim
apt install cmake build-essential
./orbit.sh --install
./orbit.sh --extra
```

3. **Install this Orbit Extension**
```js
cd ..
git clone https://github.com/andrebf6/orbit.DRL-for-Active-Semantic-SLAM.git
cd orbit.DRL-for-Active-Semantic-SLAM/
git checkout --track origin/low_level_position_controller
```


*TO DO: Setup a Dockerfile*

## Usage

### Headless mode
```js
/orbit/orbit.sh -p ./source/test_scripts/env_setup_tests/drone_efforts_base_env.py --headless
```

### Using the NVIDIA Streaming Client
```js
/orbit/orbit.sh -p ./source/test_scripts/env_setup_tests/drone_efforts_base_env.py --livestream=1
```