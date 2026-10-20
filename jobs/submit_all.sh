#!/bin/bash
# Submit Stage A, then Stage B + C for every side x method x depth mode,
# chained with Slurm dependencies.
# Usage: bash jobs/submit_all.sh [microns|v1dd] [xyz|cylindrical]
set -euo pipefail
cd /mmfs1/gscratch/amath/zihan-zhang/spatial/demo/mixture_celltype
DATASET=${1:-microns}
COORDS=${2:-xyz}
# Stage A is shared by every combination; skip it when its outputs already exist.
if [[ -f output/$DATASET/clouds_axon.npz && -f output/$DATASET/clouds_dendrite.npz ]]; then
  dependency=""
  echo "extract: reusing existing clouds in output/$DATASET"
else
  extract=$(sbatch --parsable --export=ALL,DATASET=$DATASET jobs/extract.slurm)
  dependency="--dependency=afterok:$extract"
  echo "extract: $extract"
fi
for side in axon dendrite; do
  for method in points shared_gmm; do
    for depth in relative absolute; do
      vars="ALL,DATASET=$DATASET,SIDE=$side,METHOD=$method,COORDS=$COORDS,DEPTH=$depth"
      dist=$(sbatch --parsable $dependency --export=$vars jobs/distances.slurm)
      clus=$(sbatch --parsable --dependency=afterok:$dist --export=$vars jobs/cluster.slurm)
      echo "$side/$method/$COORDS/$depth: distances $dist, cluster $clus"
    done
  done
done
