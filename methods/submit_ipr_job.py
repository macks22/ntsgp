import os
import logging
import argparse
import subprocess as sub

from ipr_runner import IPR, make_ipr_parser
from scaffold import setup


SCRIPT = """
#!/bin/bash

# Job name
#$ -N {name}

# Use cwd for submitting and writing output
#$ -cwd

# Parallelization settings
#$ -pe shared {njobs}

# Memory per slot
#$ -l mf={mem}G

# Send email to my gmu account at start and finish
#$ -M msweene2@gmu.edu
#$ -m be

# Load necessary modules
source /etc/profile
module load gcc/4.8.4
module load intel-itac/8.1.4/045

# Start the job
/home/msweene2/anaconda/bin/python \
/home/msweene2/ers-data/methods/ipr_runner.py \
{data_file} \
-k {nmodels} \
-lw {lambda_w} \
-lb {lambda_b} \
-lr {lrate} \
-i {iters} \
-e {epsilon} \
-s {init_std} \
-f {feature_guide} \
{cold_start} \
-v 1

"""

def submit_script(path, job_dir, dry):
    cmd = ' '.join(['qsub', script_name])
    if dry:
        print(cmd)
        return

    # change to new directory before submitting
    curdir = os.getcwd()
    os.chdir(job_dir)

    logging.info(cmd)
    proc = sub.Popen(args, stdout=sub.PIPE)  # spawn subprocess for qsub
    retcode = proc.wait()  # wait for qsub to finish submitting

    os.chdir(curdir) # restore previous working directory
    if retcode:
        raise SystemError("job submission failed for %s" % script_name)


def make_suffix(args):
    """Create IPR model from cmdline args in order to use instance method
    for argument suffix construction.
    """
    model = IPR(1)
    for attr, val in args._get_kwargs():
        if hasattr(model, attr):
            setattr(model, attr, val)

    # use the IPR suffix minus periods for the job name and dirname
    suffix = model.args_suffix.replace('.', '')
    return suffix


def make_fname_abbrev(path):
    dirname = os.path.basename(os.path.splitext(path)[0])
    return ''.join([p[:3] for p in dirname.split('-')])


def which_dataset(path):
    """Return the dataset name depending on the file name.
    nt: non-transfer
    tr: transfer
    all: combined
    """
    return ('all' if 'all' in path else
            'tr' if 'tr' in path else
            'nt')


JOB_DIR = '/home/msweene2/ers-data/methods/jobs'
def submit_job(args):
    data_path = os.path.abspath(args.data_file)

    # create job name from filename and args
    parts = ['ipr', make_suffix(args), make_fname_abbrev(data_path)]

    if args.cs:
        parts.append('c')
        cs_str = '-c'
    else:
        cs_str = ''

    parts.append(which_dataset(data_path))
    job_name = '-'.join(parts)


    # create new directory to run job in
    job_dir = os.path.join(JOB_DIR, job_name)
    try: os.mkdir(job_dir)
    except OSError: pass

    script_name = os.path.join(job_dir, 'runscript.sh')
    script = SCRIPT.format(
        name=job_name, mem=mem, njobs=njobs,
        data_file=data_path,
        nmodels=args.nmodels,
        lambda_w=args.lambda_w,
        lrate=args.lrate,
        iters=args.iters,
        epsilon=args.epsilon,
        init_std=args.init_std,
        feature_guide=args.feature_guide,
        cold_start=cs_str)

    with open(script_name, 'w') as f:
        f.write(script)

    submit_script(script_name, job_dir, dry)


def make_parser():
    parser = make_ipr_parser()
    parser.add_argument(
        '-nj', '--njobs',
        type=int, default=4,
        help='number of processes to run methods with')
    parser.add_argument(
        '-gb', type=int, default=8,
        help='number of GB of memory to allocate for each method')
    parser.add_argument(
        '--dry', action='store_true', default=False,
        help='do not actually submit, just display qsub command')
    return parser


if __name__ == "__main__":
    args = setup(make_parser)
    data_path = os.path.abspath(args.data_file)
    logging.info('running methods on: %s' % data_path)
    submit_job(args)
