export MUJOCO_PY_MUJOCO_PATH=/data/$(whoami)/mujoco200_linux
export MUJOCO_PY_MJKEY_PATH=/data/$(whoami)/mujoco_getid

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_PY_MUJOCO_PATH/bin
nv_version_long=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
nv_version=${nv_version_long:0:3}
echo "Nvidia version: ${nv_version}."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-$nv_version
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-$nv_version/libGL.so

