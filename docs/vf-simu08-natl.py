#!/usr/bin/env python

"""
PBS + DASK
Full 2008-2018 simulation using the 'glorys-natl' velocity field
Here we use a kernel to modify Argo float configuration in the Gulf Stream Extension

If this script is called with the '--cfg' option: we launch a simulation using a float configuration file,
otherwise we submit N jobs that will launch N simulations using N pre-defined list of configuration files.
"""

import os
import sys
import pandas as pd
import xarray as xr
import numpy as np
from datetime import timedelta
from distributed import Client
import time
from subprocess import Popen, PIPE
import logging
from ifremer_utilities import load_velocity, get_a_run_template, load_json, save_json
import argparse
import json
import collections

########################
# USER-DEFINED SECTION #
########################
# Parent folder of the VirtualFleet and VirtualFleet_GulfStream repositories:
EUROARGODEV = os.path.expanduser('~/git/github/euroargodev')

# Localise the simulations database json file, assuming a standard folder structure (unchanged from github repo):
DBFILE = os.path.sep.join([EUROARGODEV, "VirtualFleet_GulfStream", "data", "simulations_db.json"])

# Output folder for simulation results:
OUTPUT_FOLDER = '/home1/datahome/gmaze/scratch/Projects/EARISE/WP2.3/Gulf_Stream_sim/simu08'

# Simulation parameters:
RUN = get_a_run_template(DBFILE)
RUN['pbs']['submit']['queue'] = 'mpi'
RUN['pbs']['submit']['walltime'] = '48:00:00'
RUN['pbs']['submit']['mem'] = '60g'
RUN['execute']['duration'] = "days=365*10"
# RUN['execute']['duration'] = "days=30"
RUN['execute']['step'] = "minutes=5"
RUN['execute']['record'] = "seconds=3600/2"
RUN['execute']['dask_hpcconfig'] = {'name': 'datarmor-local'}
# RUN['execute']['dask_hpcconfig'] = {name': 'datarmor-local', "overrides": {"cluster.n_workers": 56}}
# RUN['execute']['dask_hpcconfig'] = {'name': 'datarmor-local', "overrides": {"cluster.n_workers": 2}}

# Make a list of float configuration files to launch job with:
cfg_root = os.path.join(EUROARGODEV, "VirtualFleet_GulfStream", "data")
cfg_list = [
    # 'vf-floatcfg-default-N300',                                # run-613257
    # 'vf-floatcfg-gse-experiment-N300-5days',                   # run-614790
    # 'vf-floatcfg-gse-experiment-N300-10days-1500drift',        # run-653526
    # 'vf-floatcfg-gse-experiment-N300-5days-ASTdef',            # run-691822
    # 'vf-floatcfg-gse-experiment-N300-10days-1500drift-ASTdef', # run-711242
    # 'vf-floatcfg-gse-experiment-N300-5days-1500drift',         # run-1009213
    # 'vf-floatcfg-gse-experiment-N300-5days-GSEext',            # run-1009214
    # 'vf-floatcfg-gse-experiment-N300-5days-1500drift-GSEext',  # run-1009216
    # 'vf-floatcfg-gse-experiment-N300-10days-1500drift-GSEext', # run-1010801
    'vf-floatcfg-gse-experiment-N300-5days-1500drift-ASTdef',  # run-1132757
    'vf-floatcfg-gse-experiment-N300-5days-500drift',  # run-1132758
]
CFG_FILE_LIST = [os.path.join(cfg_root, "%s.json" % cfg_file) for cfg_file in cfg_list]

# Where is the VirtualFleet library ?
sys.path.insert(0, os.path.join(EUROARGODEV, "VirtualFleet"))

# Where is the dask-hpcconfig LOPS library ?
sys.path.insert(0, os.path.join(EUROARGODEV, "../umr-lops/dask-hpcconfig"))

########################
# Import the dask-hpcconfig LOPS library
import dask_hpcconfig

# Import the VirtualFleet library
from virtualargofleet import VelocityField, VirtualFleet, FloatConfiguration
from virtualargofleet.utilities import simu2csv

# Setup a logger for debug:
logging.getLogger("matplotlib").setLevel(logging.ERROR)
logging.getLogger("parso").setLevel(logging.ERROR)
logging.getLogger("pyproj").setLevel(logging.ERROR)
DEBUGFORMATTER = '%(asctime)s [%(levelname)s] [%(name)s] %(filename)s:%(lineno)d: %(message)s'
logging.basicConfig(
    level=logging.DEBUG,
    format=DEBUGFORMATTER,
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[logging.FileHandler("logs/logger-%s.log" % (os.getenv('PBS_JOBID') if os.getenv('PBS_JOBID') else "nojob"), mode='w')]
)
log = logging.getLogger("script")


def updateRUN_floatcfg(cfg_file):
    """Add float configurations to the global RUN db entry"""

    # Argo-float configuration:
    cfg = FloatConfiguration(cfg_file)
    RUN['argo-configuration']['file'] = cfg_file
    RUN['argo-configuration']['name'] = cfg.name

    # Load the default config for comparison, we only record in RUN the new or modified parameters:
    cfg_default = FloatConfiguration(cfg.name)
    for p in cfg.params:
        if p in cfg_default.params and cfg.mission[p] != cfg_default.mission[p]:
            RUN['argo-configuration'][p] = cfg.mission[p]
        elif p not in cfg_default.params:
            RUN['argo-configuration'][p] = cfg.mission[p]


def updateRUN_pbsinfo(cfg_file):
    """Add some PBS information to the global RUN db entry"""
    jobid = os.getenv('PBS_JOBID').split('.')[0] if os.getenv('PBS_JOBID') else "nojob"

    # PBS data:
    RUN['pbs']['jobid'] = jobid
    RUN['pbs']['log']['pbs'] = "${scratch}/logs/Virt_Fleet.o%s" % jobid  # PBS standard output (printed once job finished)
    RUN['pbs']['log']['debug'] = "${scratch}/logs/logger-%s.log" % os.getenv('PBS_JOBID')  # script logging
    RUN['pbs']['log']['script'] = "${scratch}/logs/stdout-%s.log" % os.getenv('PBS_JOBID')  # script standard output

    # RUN['pbs']['submit']['file'] = "${repo}/local_work/vf-simu08-natl.py --cfg %s" % RUN['argo-configuration']['file']
    RUN['pbs']['submit']['file'] = os.path.join(OUTPUT_FOLDER, "vf-simu-natl-%s.pbs" % jobid)
    job_string = get_a_pbs_string(cfg_file)
    with open(RUN['pbs']['submit']['file'], 'w') as f:
        f.writelines(job_string)


def client_info(cl):
    result = {'processes': '', 'threads': '', 'memory': '', 'dashboard': cl.dashboard_link}
    info = cl._scheduler_identity
    addr = info.get("address")
    if addr:
        workers = info.get("workers", {})
        result['processes'] = len(workers)
        result['threads'] = sum(w["nthreads"] for w in workers.values())
        result['memory'] = "%0.2f Gib" % float(sum([w["memory_limit"] for w in workers.values()])/1024/1024/1024)
    return result


def sizeof_fmt(num, suffix="b"):
    # for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
    for unit in ["", "k", "m", "g", "t"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0


def updateRUN_output(VFleet):
    # Re-name simulation output file with job id:
    if os.getenv('PBS_JOBID'):
        jobid = os.getenv('PBS_JOBID').split('.')[0] if os.getenv('PBS_JOBID') else "nojob"
        output_file = os.path.join(OUTPUT_FOLDER, "vf-simu-natl-%s.nc" % jobid)
        os.rename(VFleet.run_params['output_file'], output_file)
        RUN['output']['file'] = output_file
        log.debug("Renamed simulation file '%s' to '%s'" % (VFleet.run_params['output_file'], output_file))

        RUN['output']['size-on-disk'] = sizeof_fmt(os.stat(output_file).st_size)
        ds = xr.open_dataset(output_file, engine='netcdf4')
        RUN['output']['array-size'] = {'traj': len(ds['traj']), 'obs': len(ds['obs'])}


def launch_sim(cfg_file):
    """Launch a Virtual Fleet simulation

    Uses global RUN information
    """
    jobid = os.getenv('PBS_JOBID').split('.')[0] if os.getenv('PBS_JOBID') else "nojob"

    if 'overrides' in RUN['execute']['dask_hpcconfig']:
        overrides = RUN['execute']['dask_hpcconfig']['overrides']
        cluster = dask_hpcconfig.cluster(RUN['execute']['dask_hpcconfig']['name'], **overrides)
    else:
        cluster = dask_hpcconfig.cluster(RUN['execute']['dask_hpcconfig']['name'])
    client = Client(cluster)
    RUN['execute']['dask_hpcconfig']['client'] = client_info(client)
    log.warning(client)

    # Import deployment plan
    # (update RUN if changed from template)
    df_plan = pd.read_json(os.path.join(EUROARGODEV, 'VirtualFleet_GulfStream/data/2008-2018-Deployment-Plan.json'), orient='records')

    # Load velocity field
    # (update RUN if changed from template)
    log.debug("<Virtual Fleet Velocity Field><start>")
    start_execution = time.time()  # Simulation time tracking
    VELfield = load_velocity(name='GLORYS-NATL')
    log.debug("Total execution time: %f seconds" % (time.time()-start_execution))
    log.debug("<Virtual Fleet Velocity Field><end>")

    # Load Argo floats configuration for this run:
    cfg = FloatConfiguration(cfg_file)
    log.warning(cfg)

    # Save the Argo float configuration for this run:
    runcfg_file = os.path.join(OUTPUT_FOLDER, "vf-simu-natl-%s_floatcfg.json" % jobid)
    cfg.to_json(runcfg_file)

    # Set-up a Virtual Fleet:
    log.debug("<Virtual Fleet Simulation><setup>")
    VFleet = VirtualFleet(lat=df_plan['latitude'],
                          lon=df_plan['longitude'],
                          time=np.array([np.datetime64(t) for t in df_plan['date'].dt.strftime('%Y-%m-%d %H:00').array]),
                          vfield=VELfield,
                          mission=cfg.mission,
                          verbose_events=False)

    # Run simulation:
    log.debug("<Virtual Fleet Simulation><start>")
    saveRUN_to_simdb()  # Update database with ongoing simulation just before launch ...
    start_execution = time.time()  # Simulation time tracking
    VFleet.simulate(duration=eval("timedelta(%s)" % RUN['execute']['duration']),
                    step=eval("timedelta(%s)" % RUN['execute']['step']),
                    record=eval("timedelta(%s)" % RUN['execute']['record']),
                    output_folder=OUTPUT_FOLDER,
                    verbose_progress=True,
                    )
    log.debug("Total simulation execution time: %f seconds" % (time.time()-start_execution))
    log.debug("<Virtual Fleet Simulation><end>")

    # Compute and save Argo profiles index file:
    log.debug("<Argo profiles index file><start>")
    start_execution = time.time()  # Index creation time tracking
    simu_file = VFleet.run_params['output_file']
    if os.getenv('PBS_JOBID'):
        index_file = os.path.join(OUTPUT_FOLDER, "vf-simu-natl-%s_ar_index_prof.txt" % jobid)
    else:
        index_file = VFleet.run_params['output_file'].replace(".nc", "_floatcfg.json")
    simu2csv(simu_file, index_file)
    RUN['output']['index'] = index_file
    log.debug("Execution time: %f seconds" % (time.time()-start_execution))
    log.debug("<Argo profiles index file><end>")

    #
    updateRUN_output(VFleet)

    cluster.close()
    return True


def saveRUN_to_simdb():
    """Add more information to the global RUN db entry and save on disk the updated database file"""
    jobid = os.getenv('PBS_JOBID').split('.')[0] if os.getenv('PBS_JOBID') else "nojob"

    # Load database file to update:
    db = load_json(DBFILE)

    # Add, or over-write, this new run to the database:
    db['simulations']["run-%s" % jobid] = RUN

    # Re-order runs:
    db['simulations'] = collections.OrderedDict(sorted(db['simulations'].items(), reverse=True))

    # Update database file:
    save_json(db, db_file=DBFILE)
    # save_json(db, db_file="%s.json" % os.getenv('PBS_JOBID'))


def get_a_pbs_string(cfg_file):
    # Customize PBS options here or in the RUN global dict
    queue = RUN['pbs']['submit']['queue']
    walltime = RUN['pbs']['submit']['walltime']
    mem = RUN['pbs']['submit']['mem']

    job_string = """#!/bin/bash
#PBS -N Virt_Fleet
#PBS -q %s
#PBS -l walltime=%s
#PBS -l mem=%s
# PBS -l ncpus=28
#PBS -j oe
cd $PBS_O_WORKDIR
qstat -f $PBS_JOBID
pbsnodes $HOST
source /usr/share/Modules/3.2.10/init/bash
module load conda/latest
source activate /home1/datahome/gmaze/conda-env/virtualfleet
logfile="/home1/datahome/gmaze/scratch/Projects/EARISE/WP2.3/Gulf_Stream_sim/logs/stdout-${PBS_JOBID}.log"
./vf-simu08-natl.py --cfg %s >& $logfile
    """ % (queue, walltime, mem, cfg_file)

    return job_string


def submit_job(cfg_file):
    """Submit a PBS job to execute a simulation, given an Argo float configuration file"""
    job_string = get_a_pbs_string(cfg_file)

    # Send job_string to qsub
    proc = Popen(["qsub", job_string], shell=True, close_fds=True, stdin=PIPE, stdout=PIPE)

    if (sys.version_info > (3, 0)):
        proc.stdin.write(job_string.encode('utf-8'))
    else:
        proc.stdin.write(job_string)
    out, err = proc.communicate()

    # Print your job and the system response to the screen as it's submitted
    # print(job_string)
    print(out)

    time.sleep(0.1)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='VirtualFleet simulation launcher',
                                     epilog="(c) LOPS/Ifremer, 2022")

    # Add long and short arguments
    parser.add_argument("--cfg", help="Argo float configuration file name", default=None)

    # If this script is called with the 'cfg' option, we launch a simulation, otherwise we submit PBS jobs

    # Read arguments from the command line
    args = parser.parse_args()
    if args.cfg:
        # Launch a simulation with this float configuration file:
        updateRUN_floatcfg(args.cfg)
        updateRUN_pbsinfo(args.cfg)
        launch_sim(args.cfg)  # this will update the database with the ongoing simulation just before simulation launch
        saveRUN_to_simdb()
    else:
        for cfg_file in CFG_FILE_LIST:
            submit_job(cfg_file)
