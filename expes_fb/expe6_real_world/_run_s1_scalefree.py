"""Wrapper: full Setting-1 re-run under the scale-free band, to a tagged results dir,
with the oracle-isolated null_tr_wl1 arm added in run_s1."""
import sys; sys.path.insert(0, '.')
from pathlib import Path
import run_s1 as S1
S1.RESULTS_DIR = S1.RESULTS_DIR.parent / 'setting1_scalefree'
S1.RESULTS_PATH = S1.RESULTS_DIR / 'results.pkl'
S1.CHECKPOINT_PATH = S1.RESULTS_DIR / 'results_checkpoint.pkl'
S1.CHECKPOINT_META_PATH = S1.RESULTS_DIR / 'results_checkpoint_meta.json'
S1.main()
