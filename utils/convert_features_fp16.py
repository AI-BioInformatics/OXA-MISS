"""
float16 .npy copies of the UNIv2 WSI features (one file per slide), read memory-mapped by the dataloader
(dataloader_multidataset.load_wsi_bag): the jobs of a node share the features through the page cache instead of
each holding a private ~200 GB RAM copy.

Every slide is checked before writing (CLAM features .h5 vs the patches .h5 of patches_CLAM):
    rows of 'features' == rows of 'coords' of the features .h5 == rows of 'coords' of the patches .h5
    coords identical and in the same order (the features of patch i are those of patch i)
    every value finite, also after the float16 cast (|x| < 65504)
A slide that fails is not written and is listed in the report. Writing goes through <slide>.npy.tmp + rename
(an interrupted job leaves no partial .npy); a rerun skips the slides whose .npy exists with the right shape.

Cohorts: config/<TCGA_* | CPTAC_*>_dataset.yaml; source = <H5_ROOT>/<cohort>/features_UNIv2/h5_files,
destination = /work/H2020DeciderFicarra/<TCGA|CPTAC>/<cohort>/features_UNIv2_fp16 (quota of H2020DeciderFicarra).

Usage (reads ~620 GB: run it as a SLURM job, batch/convert_features_fp16.sbatch):
    python utils/convert_features_fp16.py [--datasets TCGA_KIRC CPTAC_UCEC] [--workers 16]
"""
import argparse
import glob
import os
import sys
import time
from multiprocessing import Pool

import h5py
import numpy as np
import pandas as pd
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST_ROOT = '/work/H2020DeciderFicarra'
FP16_MAX = float(np.finfo(np.float16).max)


# cohort folders of the CLAM outputs (features_UNIv2/h5_files, patches_CLAM/patches) per source; independent of
# the yamls' pt_files_path, which points to the float16 copies once they exist
H5_ROOT = {'TCGA': '/work/h2020deciderficarra_shared/TCGA', 'CPTAC': '/work/H2020DeciderFicarra/CPTAC'}


def cohort_paths(cfg_path):
    """(name, features .h5 folder, patches .h5 folder, destination folder) of a dataset yaml."""
    name = yaml.safe_load(open(cfg_path))['name']
    source, cohort = name.split('_', 1)
    cohort_dir = os.path.join(H5_ROOT[source], cohort)
    return name, os.path.join(cohort_dir, 'features_UNIv2', 'h5_files'), os.path.join(cohort_dir, 'patches_CLAM', 'patches'), \
        os.path.join(DEST_ROOT, source, cohort, 'features_UNIv2_fp16')


def convert(args):
    h5_path, patches_dir, dest_dir = args
    slide = os.path.basename(h5_path)[:-len('.h5')]
    out = os.path.join(dest_dir, slide + '.npy')
    try:
        with h5py.File(h5_path, 'r') as f:
            n_feat, dim = f['features'].shape
            n_coords = f['coords'].shape[0]
            if os.path.exists(out):
                done = np.load(out, mmap_mode='r')
                if done.shape == (n_feat, dim) and done.dtype == np.float16:
                    return slide, 'skipped (done)', n_feat
            feats = f['features'][:]
            coords = f['coords'][:]
        patch_h5 = os.path.join(patches_dir, slide + '.h5')
        if not os.path.exists(patch_h5):
            return slide, 'no patches .h5', n_feat
        with h5py.File(patch_h5, 'r') as f:
            patch_coords = f['coords'][:]
        if not (n_feat == n_coords == len(patch_coords)):
            return slide, f'rows: features {n_feat}, feature coords {n_coords}, patches {len(patch_coords)}', n_feat
        if not np.array_equal(coords.astype(np.int64), patch_coords.astype(np.int64)):
            return slide, 'coords differ from patches_CLAM (order or values)', n_feat
        if not np.isfinite(feats).all():
            return slide, 'non-finite features', n_feat
        if np.abs(feats).max() >= FP16_MAX:
            return slide, f'|value| {np.abs(feats).max():.1f} overflows float16', n_feat
        half = feats.astype(np.float16)
        tmp = out + '.tmp'
        with open(tmp, 'wb') as fh:
            np.save(fh, half)
        os.replace(tmp, out)
        return slide, 'ok', n_feat
    except Exception as e:   # unreadable / truncated file
        return slide, f'error: {type(e).__name__}: {e}'[:200], -1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--datasets', nargs='+', default=None, help='dataset names; default: every TCGA_ / CPTAC_ yaml')
    ap.add_argument('--workers', type=int, default=16)
    args = ap.parse_args()
    cfgs = sorted(glob.glob(os.path.join(REPO, 'config', 'TCGA_*_dataset.yaml')) +
                  glob.glob(os.path.join(REPO, 'config', 'CPTAC_*_dataset.yaml')))
    old_umask = os.umask(0o007)
    report = []
    for cfg in cfgs:
        name, src, patches_dir, dest = cohort_paths(cfg)
        if args.datasets and name not in args.datasets:
            continue
        os.makedirs(dest, exist_ok=True)
        files = sorted(glob.glob(os.path.join(src, '*.h5')))
        t0 = time.time()
        with Pool(args.workers) as pool:
            res = pool.map(convert, [(f, patches_dir, dest) for f in files], chunksize=4)
        status = pd.Series([r[1] for r in res])
        ok = status.isin(['ok', 'skipped (done)']).sum()
        print(f'{name:12s} {len(files):5d} slides | ok {ok} | problems {len(files) - ok} | {time.time() - t0:.0f}s -> {dest}',
              flush=True)
        for slide, st, n in res:
            report.append({'dataset': name, 'slide_id': slide, 'status': st, 'n_patches': n,
                           'source': os.path.join(src, slide + '.h5'), 'npy': os.path.join(dest, slide + '.npy')})
            if st not in ('ok', 'skipped (done)'):
                print(f'   PROBLEM {slide}: {st}', flush=True)
    os.umask(old_umask)
    out = os.path.join(DEST_ROOT, 'TCGA', 'features_UNIv2_fp16_conversion_report.csv')
    pd.DataFrame(report).to_csv(out, index=False)
    print(f'report: {out} ({sum(r["status"] not in ("ok", "skipped (done)") for r in report)} problems)')


if __name__ == '__main__':
    sys.exit(main())
