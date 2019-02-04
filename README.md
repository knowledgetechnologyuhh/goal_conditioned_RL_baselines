Getting started

1. Install MuJoCo (mujoco.org) and copy mjpro150 folder as well as mjkey.txt to ~/.mujoco
2. Set environment variables according to your Graphics driver version as in '''set_paths.sh'''
3. Set up virtual environment using '''virtualenv -p python3 venv'''
4. Activate virtualenvironment using '''source venv/bin/activate'
5. Install python libraries using '''pip3 install -r requirements_gpu.txt'''
6. Run script with '''experiment/train.py'''



== Currently supported algorithms ==
Algorithm-specific implementation details are stored in '''baselines/<alg name>'''. 
We currently support '''baselines.her''' as comparison and baseline to our results. We implement '''baselines.model_based'''.

== Command line options ==

General command line options can be found in '''experiment/click_options.py'''

Algoritm specific command line options can be found in ```baselines/<alg name>/interface/click_options.py'''

