#For MuJoCo
export MUJOCO_PY_MUJOCO_PATH=/data/$(whoami)/mujoco200_linux
export MUJOCO_PY_MJKEY_PATH=/data/$(whoami)/mujoco_getid/mjkey.txt
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_PY_MUJOCO_PATH/bin

#For CoppeliaSim
export COPPELIASIM_ROOT=/data/$(whoami)/CoppeliaSim_Edu_V4_0_0_Ubuntu18_04
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

#For both
nv_version_long=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
nv_version=${nv_version_long:0:3}
echo "Nvidia version: ${nv_version}."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-$nv_version
export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so
