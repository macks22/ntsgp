import os
import logging
import argparse
import subprocess as sub


SCRIPT = """
#!/bin/bash

# Job name
#$ -N libfm-{name}

# Use cwd for submitting and writing output
#$ -cwd

# Memory per slot
#$ -l mf=8G

# Send email to my gmu account at start and finish
#$ -M msweene2@gmu.edu
#$ -m be

# Load necessary modules
source /etc/profile
module load gcc/4.8.4
module load intel-itac/8.1.4/045

# Start the job
/home/msweene2/anaconda/bin/python \
/home/msweene2/ers-data/methods/libfm.py \
/home/msweene2/ers-data/data/preprocessed-data.csv \
-d {dim} -i {iter} -s {std} -b \
{features}
"""

JOB_DIR = '/home/msweene2/ers-data/methods/jobs'
def submit_job(name, dim, iter, std, features, dry=False):
    job_name = '-'.join([
        name,
        'd%d' % dim,
        'i%d' % iter,
        's%s' % ''.join(str(std).split('.'))
    ])

    job_dir = os.path.join(JOB_DIR, job_name)
    try: os.mkdir(job_dir)
    except OSError: pass

    script_name = os.path.join(job_dir, 'runscript.sh')
    script = SCRIPT.format(
        name=name, dim=dim, iter=iter, std=std, features=features)

    with open(script_name, 'w') as f:
        f.write(script)

    curdir = os.getcwd()
    os.chdir(job_dir)

    args = ['qsub', script_name]
    cmd = ' '.join(args)
    if dry:
        print cmd
    else:
        logging.info(cmd)
        proc = sub.Popen(args, stdout=sub.PIPE)
        retcode = proc.wait()
        if retcode:
            os.chdir(curdir)
            raise SystemError("job submission failed for %s" % script_name)

    os.chdir(curdir)


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-n', '--name',
        default='job')
    parser.add_argument(
        '-d', '--dim',
        type=int, default=4)
    parser.add_argument(
        '-i', '--iter',
        type=int, default=200)
    parser.add_argument(
        '-s', '--std',
        type=float, default=0.1)
    parser.add_argument(
        '-f', '--features',
        action='store', default='')
    parser.add_argument(
        '--dry', action='store_true', default=False)
    parser.add_argument(
        '-v', '--verbose',
        action='store_true', default=False)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s]: %(message)s')

    submit_job(args.name, args.dim, args.iter, args.std, args.features,
               args.dry)
