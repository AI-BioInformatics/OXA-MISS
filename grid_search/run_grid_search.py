"""
Submits a grid search as one SLURM array job: task i runs main.py on version i of the versions json
(see grid_search/make_modality_grid.py). Run it on the login node: it only calls sbatch.

    python grid_search/run_grid_search.py --config config/OXA_MISS_ccRCC.yaml \
        --versions grid_search/models_versions/OXA_MISS_ccRCC_modalities.json [--max_parallel 8] [--dry_run]
        [--sbatch_args="--mem=120G --time=24:00:00"]
    extra main.py arguments after --, e.g.  -- --seed 43

Two people launching the same versions file: each takes one part, --part 1/2 and --part 2/2 (version i goes to
part (i mod N) + 1: every part gets about the same share of each variant and seed). Both must see the same
"versions ... sha256 ..." line: the file must not be regenerated in between.

Runs stopped by the 24 h limit: once the array is over, the same command with --resume submits again only the
versions of the part without a finished run (RUN_COMPLETE in its results folder), with main.py --resume: the
folds already trained are evaluated from their checkpoints, the interrupted fold is retrained.
"""
import argparse
import glob
import hashlib
import json
import os
import shlex
import subprocess
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SH_PATH = os.path.join(REPO, 'grid_search', 'run_grid_search.sh')


def parse_part(text):
    try:
        k, n = (int(x) for x in text.split('/'))
    except ValueError:
        raise SystemExit(f"--part {text}: expected K/N, e.g. 1/2")
    if not 1 <= k <= n:
        raise SystemExit(f"--part {text}: K must be in 1..N")
    return k, n


def finished(project_dir, version):
    """True if the version's results folder has a run of its seed with RUN_COMPLETE (main.py, end of the run)."""
    experiment_dir = os.path.join(version.get('project_dir', project_dir), version.get('results_subdir') or '',
                                  version.get('title', ''))
    for cfg_path in glob.glob(os.path.join(experiment_dir, '*', 'config.yaml')):
        if not os.path.exists(os.path.join(os.path.dirname(cfg_path), 'RUN_COMPLETE')):
            continue
        with open(cfg_path) as f:
            if (yaml.safe_load(f) or {}).get('seed') == version.get('seed'):
                return True
    return False


def compress(indices):
    """[0, 1, 2, 5, 7, 8] -> '0-2,5,7-8' (sbatch --array syntax)."""
    out, start = [], None
    for i, x in enumerate(indices):
        if start is None:
            start = x
        if i + 1 == len(indices) or indices[i + 1] != x + 1:
            out.append(str(start) if start == x else f"{start}-{x}")
            start = None
    return ','.join(out)


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index('--') + 1:] if '--' in argv else []
    argv = argv[:argv.index('--')] if '--' in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True)
    ap.add_argument('--versions', required=True)
    ap.add_argument('--indices', default=None, help='subset of versions, sbatch --array syntax (e.g. 0,3,7-9)')
    ap.add_argument('--part', default=None, help='K/N: only the versions i with i mod N == K-1 (two people: 1/2 and 2/2)')
    ap.add_argument('--resume', action='store_true',
                    help='only the versions (of --part) without a finished run, with main.py --resume')
    ap.add_argument('--max_parallel', type=int, default=8, help='array tasks running at the same time')
    ap.add_argument('--sbatch_args', default='', help='extra sbatch options (override the #SBATCH of '
                    'run_grid_search.sh); use the = form, e.g. --sbatch_args=--mem=120G or '
                    '--sbatch_args="--mem=120G --time=24:00:00" (a value starting with -- would be read as an option)')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args(argv)
    if args.part and args.indices:
        raise SystemExit("--part and --indices are alternatives")

    config, versions_path = os.path.abspath(args.config), os.path.abspath(args.versions)
    for path in (config, versions_path, SH_PATH):
        if not os.path.isfile(path):
            raise SystemExit(f"not found: {path}")
    with open(versions_path, 'rb') as f:
        raw = f.read()
    versions = json.loads(raw)
    n = len(versions)
    if n == 0:
        raise SystemExit(f"{versions_path} has no versions")
    print(f"versions {os.path.basename(versions_path)}: {n} | sha256 {hashlib.sha256(raw).hexdigest()[:12]}")

    if args.part:
        k, parts = parse_part(args.part)
        selected = [i for i in range(n) if i % parts == k - 1]
    elif args.indices:
        selected = None   # passed through as given
    else:
        selected = list(range(n))
    if args.resume:
        if selected is None:
            raise SystemExit("--resume works with --part or with the whole file, not with --indices")
        name = f"ABL_{os.path.splitext(os.path.basename(versions_path))[0]}"
        queued = subprocess.run(['squeue', '-h', '-u', os.environ.get('USER', ''), '-n', name, '-o', '%i'],
                                capture_output=True, text=True).stdout.split()
        if queued:
            raise SystemExit(f"jobs of {name} still queued / running ({queued[:3]}): resume after they end")
        with open(config) as f:
            project_dir = yaml.safe_load(f)['project_dir']
        selected = [i for i in selected if not finished(project_dir, versions[i])]
        print(f"--resume: {len(selected)} versions without a finished run")
        if not selected:
            return
        extra = ['--resume'] + extra
    indices = args.indices or compress(selected)
    print(f"array indices: {indices} ({len(selected) if selected is not None else 'given'} runs)")
    job_name = f"ABL_{os.path.splitext(os.path.basename(versions_path))[0]}"
    cmd = ['sbatch', f'--array={indices}%{args.max_parallel}', f'--job-name={job_name}'] + \
          shlex.split(args.sbatch_args) + [SH_PATH, config, versions_path] + extra
    print(' '.join(cmd))
    if not args.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
