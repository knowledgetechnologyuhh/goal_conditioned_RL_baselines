All CoppeliaSim environment-names have to start with 'Cop'. 
They do not implement a render function, because you can look at the simulation when you don't run in headless mode,
so you can still choose whether to see the environment with the `--render` argument.
The `CopReacherEnv` receives an argument ik (0 or 1), with which you can choose to use inverse
kinematics. Note that sometimes, the inverse kinematics cannot be computed and the action is
not carried out.

**Installation of CoppeliaSim and PyRep**

You need CoppeliaSim and PyRep. 
Download CoppeliaSim [here](https://www.coppeliarobotics.com/ubuntuVersions) and start it to check whether it works.
Then `git clone https://github.com/stepjam/PyRep.git`. I put both in /data/*username*/, you are free to put them
somewhere else, but you'll have to adjust the paths. 

Run `set_paths.sh` and, if you want to use run or debug in PyCharm, set the full paths for 
COPPELIASIM_ROOT, LD_LIBRARY_PATH and QT_QPA_PLATFORM_PLUGIN_PATH
in your run/debug configuration, e.g.:
```
COPPELIASIM_ROOT=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
QT_QPA_PLATFORM_PLUGIN_PATH=EDIT/ME/PATH/TO/COPPELIASIM/INSTALL/DIR
```

Pip install PyRep by activating you virtual environment and running: 

`pip install git+https://github.com/stepjam/PyRep.git`

You can find some troubleshooting on the PyRep git-page.

Now you can test the installation by running
`--env CopReacherEnv-ik1-v0 --algorithm baselines.her`

**Known issues**

If the code does not execute because of the following error: `ImportError: libcoppeliaSim.so.1: cannot open shared object file: No such file or directory` Then simply navigate to your COPELIASIM_ROOT and add a symlink by executing `ln -s libcoppeliaSim.so libcoppeliaSim.so.1` 


