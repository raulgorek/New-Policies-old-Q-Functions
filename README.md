# New Policies - Old Q-Functions
This repo relys holds the code needed to extract new policies with the TD3+BC objective from CQL trained Q-Functions.

This repo is built entirely on top of the OfflineRL-Kit library. The code in the `OfflineRL-Kit_Code` folder contains slightly modified code of the [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit) library while the `code` directory contains the same pythn code but altered strongly.

To get this code to run on an M1 Mac please follow the instructions in [#682](https://github.com/openai/mujoco-py/issues/682). Otherwise you can follow the installation instructions below:
``` bash
conda create -n cql python=3.8.19 -y
pip install -r requirements.txt
```
To get the CQL models please either train them yourself by using the following command or download the pretrained models from the [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit) and put the results into the `log` directory
``` bash
python OfflineRL-Kit_Code/run_cql.py --seed <seed> --task <task>
```
The `log` directory should have the following structure:
```
log
 |-- task
       |-- algo
            |-- seed&timestamp
                |-- model
                |-- record
                |-- ...
```
were the model directory holds the model of the last training step and the record a tracker .csv file with the training curve.
Then you can launch the code either manually by launching
``` bash
python code/ddpg-bc-cql.py --load-path log --task <task e.g. hopper-medium-v2>
```
or use the provided bash scripts to execute the runs over multiple seeds, tasks and parameter configurations. Just change the parameters in the script as you need them. You can launch the scripts directly in bash from main directory.
``` bash
./code/run_experiments_ddpg_bc_cql.sh
```

To plot the results use the code provided by the [OfflineRL-Kit](https://github.com/yihaosun1124/OfflineRL-Kit) like this:
``` bash
python OfflineRL-Kit_Code/plotter.py --task <task> --algos "cql" "DDPG+BC_CQL_<put your value for alpha here>" --show-legend
```