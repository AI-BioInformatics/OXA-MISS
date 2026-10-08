"""
Submits a grid search as one SLURM array job: task i runs main.py on version i of the versions json
(see grid_search/make_modality_grid.py). Run it on the login node: it only calls sbatch.

    python grid_search/run_grid_search.py --config config/OXA_MISS_ccRCC.yaml \
        --versions grid_search/models_versions/OXA_MISS_ccRCC_modalities.json [--max_parallel 8] [--dry_run]
        [--sbatch_args="--mem=120G --time=24:00:00"]
    extra main.py arguments after --, e.g.  -- --seed 43
"""
import argparse
import json
import os
import shlex
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SH_PATH = os.path.join(REPO, 'grid_search', 'run_grid_search.sh')


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index('--') + 1:] if '--' in argv else []
    argv = argv[:argv.index('--')] if '--' in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--versions', required=True)
    ap.add_argument('--indices', default=None, help='subset of versions, sbatch --array syntax (e.g. 0,3,7-9)')
    ap.add_argument('--max_parallel', type=int, default=8, help='array tasks running at the same time')
    ap.add_argument('--sbatch_args', default='', help='extra sbatch options (override the #SBATCH of '
                    'run_grid_search.sh); use the = form, e.g. --sbatch_args=--mem=120G or '
                    '--sbatch_args="--mem=120G --time=24:00:00" (a value starting with -- would be read as an option)')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args(argv)

    config, versions = os.path.abspath(args.config), os.path.abspath(args.versions)
    for path in (config, versions, SH_PATH):
        if not os.path.isfile(path):
            raise SystemExit(f"not found: {path}")
    with open(versions) as f:
        n = len(json.load(f))
    if n == 0:
        raise SystemExit(f"{versions} has no versions")
    indices = args.indices or f"0-{n - 1}"
    cmd = ['sbatch', f'--array={indices}%{args.max_parallel}'] + shlex.split(args.sbatch_args) + \
          [SH_PATH, config, versions] + extra
    print(' '.join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
