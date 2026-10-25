# Shared job environment: mixture conda env, threads capped to the allocation.
PYTHON=/mmfs1/gscratch/amath/zihan-zhang/miniconda3/envs/mixture/bin/python
# joblib workers each run single-threaded BLAS; parallelism comes from joblib.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
# Change into the project root (the parent of this file's folder), whichever checkout this is.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
