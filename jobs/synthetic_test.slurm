#!/bin/bash
#SBATCH --job-name=celltype_test
#SBATCH --account=amath
#SBATCH --partition=cpu-g2
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=log/synthetic_test.txt
# Unit tests, then the full Stage B + C pipeline on synthetic clouds with three known shapes.
set -euo pipefail
source jobs/env.sh
$PYTHON -m pytest tests -q -p no:cacheprovider
$PYTHON -u scripts/make_synthetic.py
for coords in xyz cylindrical; do
  for method in points shared_gmm; do
    for rep in 0 1 2 3; do
      $PYTHON -u scripts/02_compute_distances.py --dataset synthetic --side axon \
        --method $method --coords $coords --rep $rep
    done
    $PYTHON -u scripts/03_cluster_neurons.py --dataset synthetic --side axon \
      --method $method --coords $coords --k-max 8
  done
done
