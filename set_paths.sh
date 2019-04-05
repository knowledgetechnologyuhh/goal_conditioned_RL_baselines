export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/informatik3/wtm/home/$(whoami)/.mujoco/mjpro150/bin

nv_version=$(modinfo $(find /lib/modules/$(uname -r) -iname nvidia_*.ko | head -1) | grep ^version: | cut -d' ' -f 9 | cut -d'.' -f 1)
echo "Nvidia version: ${nv_version}."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-$nv_version
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-$nv_version/libGL.so

#nv_version=418
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-$nv_version
#export LD_PRELOAD=$LD_PRELOAD:/usr/lib/x86_64-linux-gnu/libGLEW.so:/usr/lib/nvidia-$nv_version/libGL.so