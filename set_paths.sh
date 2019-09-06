#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mjpro150/bin
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco200/bin
export MUJOCO_PY_MUJOCO_PATH=/data/$(whoami)/mujoco200_linux
export LD_LIBRARY_PATH=MUJOCO_PY_MUJOCO_PATH/bin
export MUJOCO_PY_MJKEY_PATH=/data/$(whoami)/mujoco_getid
#nv_version=$(modinfo $(find /lib/modules/$(uname -r) -iname nvidia_*.ko | head -1) | grep ^version: | cut -d' ' -f 9 | cut -d'.' -f 1)
#echo "Nvidia version: ${nv_version}."
nv_version_long=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
nv_version=${nv_version_long:0:3}
echo "Nvidia version: ${nv_version}."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-$nv_version
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-$nv_version/libGL.so

