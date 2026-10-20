"""
Label-Free Cell-Type Discovery from Synapse Anatomy
===================================================
Library code for clustering proofread neurons by the spatial distribution of
their own synapses, without using dataset cell-type labels. Axon (outgoing)
and dendrite (incoming) sides are always handled separately.

Modules:
  config      paths, dataset tables, output layout, Slurm-aware CPU count
  clouds      extract per-neuron synapse clouds and read them back (CSR)
  distances   weighted resampling and neuron-to-neuron Wasserstein distances
  clustering  hierarchical clustering, stability, noise floor, label/confound checks
  plotting    figures for the clustering stage
  pipeline    the three stages (extract, distances, cluster)

Stages are in pipeline.py; the command-line entry point is ../scripts/run_pipeline.py.
"""
