import neurom as nm
import pandas as pd
from axon_synthesis.utils import get_morphology_paths
import seaborn as sns
from axon_projection.compute_morphometrics import compute_stats_parallel
from compute_scores import compute_stats_populations, plot_population_stats


morphometrics = ["total_length_per_neurite"]

# All
bio_LRA = get_morphology_paths("/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/morpho/all_morphs_final")["morph_path"].values.tolist()

# local axons
axons_list_file = "/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/circuit-build/lite_MOp5_final/bioname/neurondb-axon.dat"
axons_location = "/gpfs/bbp.cscs.ch/project/proj148/axon_morph_release/scaled_axons_asc/"
df_local = pd.read_csv(axons_list_file, names=['morph_name', 'layer', 'mtype'], sep=' ')
df_local['morph_path'] = df_local.apply(lambda row: axons_location + row['morph_name'] + ".asc", axis=1)
bio_local_list = df_local["morph_path"].values.tolist()

# compute stats
df_stats = compute_stats_populations(bio_LRA, bio_local_list, morphometrics, morph_type="bio", out_file="total_lengths")
df_stats = pd.read_json("total_lengths_bio.json")
df_stats.index.name = "morphometrics"
# plot
plot_population_stats(df_stats['pop_1'], df_stats['pop_2'], morphometrics, morph_type="bio", out_file="total_lengths", feat_name=r'Total length per neurite [$\mu$m]', type_1="Biological LRAs", type_2="Biological local axons")

# Do also for MOp5 only
bio_LRA_MOp5 = get_morphology_paths("/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/morpho/MOp5_final")["morph_path"].values.tolist()

# filter L5:PCs because they are the ones we use for MOp5
df_local = df_local[(df_local['layer']==5) & (df_local['mtype'].str.contains('PC'))]
bio_local_list = df_local["morph_path"].values.tolist()

# df_stats = compute_stats_populations(bio_LRA_MOp5, bio_local_list, morphometrics, morph_type="bio", out_file="total_lengths_MOp5")
df_stats = pd.read_json("total_lengths_MOp5_bio.json")
df_stats.index.name = "morphometrics"
plot_population_stats(df_stats['pop_1'], df_stats['pop_2'], morphometrics, morph_type="bio", out_file="total_lengths_MOp5", feat_name=r'Total length per neurite [$\mu$m]', type_1="Biological LRAs", type_2="Biological local axons")
