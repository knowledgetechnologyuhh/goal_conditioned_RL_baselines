# Getting started

1. Install MuJoCo (mujoco.org) and copy mjpro150 folder as well as mjkey.txt to ~/.mujoco
2. Set environment variables according to your Graphics driver version as in `set_paths.sh`
3. Set up virtual environment using `virtualenv -p python3 venv`
4. Activate virtualenvironment using `source venv/bin/activate`
5. Install python libraries using `pip3 install -r requirements_gpu.txt`
6. Run script with `experiment/train.py`

Logs will be stored in a directory according to the `--logs` command line parameter (by default `logs`). It will create a subdirecory according to the git commit id and then a subdirectory according to the number of trials the experiment with the same parameters has been performed so far. The name of this second subdirectory is determined in the `main` function of `train.py`, according to line `override_params = config.OVERRIDE_PARAMS_LIST`.



# Currently supported algorithms
Algorithm-specific implementation details are stored in `baselines/<alg name>`.
We currently support `baselines.her` (Hindsight Experience Replay) as comparison and baseline to our results. This code is about `baselines.model_based`. Hence, the folder `baselines/model_based` is where most of the coding takes place. The algorithm can be selected using the command line option `--algorithm` (see below).

# Command line options
Command line options are realized using the *click* library.
General command line options can be found in `experiment/click_options.py`

Algoritm specific command line options can be found in `baselines/<alg name>/interface/click_options.py`

---
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
---
# Main algorithm
\begin{algorithm}[H]
\DontPrintSemicolon
\SetAlgoLined
\KwResult{Write here the result}
\SetKwInOut{Input}{Input}\SetKwInOut{Output}{Output}
\Input{Write here the input}
\Output{Write here the output}
\BlankLine
\While{While condition}{
    instructions\;
    \eIf{condition}{
        instructions1\;
        instructions2\;
    }{
        instructions3\;
    }
}
\caption{While loop with If/Else condition}
\end{algorithm}