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
  distances   weighted resampling and neuron-to-neuron Wasserstein distances
  shared      shared-component mixture, per-neuron mixing weights, MW2 distances
  clustering  hierarchical clustering, stability, noise floor, label/confound checks
  templates   cylindrically constrained weighted GMM used as cluster templates
  assign      kNN and template assignment with cross-validated calibration
  plotting    figures for the clustering and assignment stages
  pipeline    the stages (extract, distances, cluster; assign is optional)

Stages are in pipeline.py; the command-line entry point is ../scripts/run_pipeline.py.
"""
