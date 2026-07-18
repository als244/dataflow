"""dataflow_training: the WORKLOAD side of the engine/workload split.

Model families (configs, block executables, bridges, registry), the
shared block/kernel libraries, lowering + planning (the dataflow_sim
dependency lives here), data streaming/packing, run drivers, and the
distributed conductor. Imports the engine (``dataflow``) only through
its public surfaces: core IR, runtime ABIs, service client, and the
resolver registry.
"""
