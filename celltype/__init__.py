"""
Label-Free Cell-Type Discovery from Synapse Anatomy
===================================================
Library code for clustering proofread neurons by the spatial distribution of
their own synapses, without using dataset cell-type labels, and optionally for
assigning every other neuron to the resulting clusters (Stage D, off by
default). Axon (outgoing) and dendrite (incoming) sides are always handled
separately.

Modules:
  config      paths, dataset tables, output layout, Slurm-aware CPU count
  clouds      extract per-neuron synapse clouds and read them back (CSR)
  distances   exact Wasserstein distances between complete synapse clouds, bootstrap replicates
  shared      shared-component mixture, per-neuron mixing weights, MW2 distances
  clustering  hierarchical clustering, stability, noise floor, label/confound checks
  templates   cylindrically constrained weighted GMM used as cluster templates
  assign      kNN and template assignment with cross-validated calibration
  plotting    figures for the clustering and assignment stages and the diagnostics
  pipeline    the stages (extract, distances, cluster; validate and assign are optional)
  neuron_gmm  per-neuron Gaussian mixtures with the weighted GMM fitter imported from ../mixture,
              and their fitting stability (mixture Wasserstein distance between repeated fits)
  split_half  disjoint, sample-matched self/between distances on a fixed neuron cohort
  split_cluster  held-out validation of the consensus clustering on disjoint synapse halves

Stages are in pipeline.py; the command-line entry point is ../scripts/run_pipeline.py.
"""
