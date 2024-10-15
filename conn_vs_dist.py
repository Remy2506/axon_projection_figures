"""Module to plot the connectivity vs distance."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from axon_projection.query_atlas import load_atlas
import voxcell
from axon_synthesis.atlas import AtlasHelper, AtlasConfig
import h5py
import seaborn as sns
from axon_projection.plot_results import set_font_size
from axon_projection.choose_hierarchy_level import find_parent_acronym, build_parent_mapping
import json
import matplotlib.colors as mcolors
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.spatial.distance import cdist

def get_border_voxels(voxel_coordinates):
    """
    Returns the coordinates of all voxels that form the border of the volume.

    Parameters:
        voxel_coordinates (np.ndarray): An array of shape (n, 3) each row represents a voxel position (x, y, z).

    Returns:
        border_voxels (np.ndarray): An array containing the coordinates of all border voxels.
    """
    # Create a set of all voxel coordinates for fast lookup
    voxel_set = set(map(tuple, voxel_coordinates))

    # Prepare a list to collect border voxels
    border_voxels = []

    # Define the 6 possible neighbor offsets in 3D space
    neighbors = np.array([
        [1, 0, 0], [-1, 0, 0],  # x-direction neighbors
        [0, 1, 0], [0, -1, 0],  # y-direction neighbors
        [0, 0, 1], [0, 0, -1]   # z-direction neighbors
    ])

    # Loop over each voxel and check if it has a neighbor outside the volume
    for voxel in voxel_coordinates:
        for neighbor in neighbors:
            neighbor_voxel = tuple(voxel + neighbor)
            # If any neighbor is not in the set, the current voxel is on the border
            if neighbor_voxel not in voxel_set:
                border_voxels.append(voxel)
                break

    return np.array(border_voxels)

def compute_min_distances(point1, set2):
    """Helper function to compute the minimum distance between a point and the second set."""
    return np.min(np.linalg.norm(set2 - point1, axis=1))

def min_distance_between_sets(set1, set2, num_workers=72):
    """
    Compute the minimum Euclidean distance between two sets of 3D points in parallel.
    
    Parameters:
    set1 (numpy array): First set of coordinates (Nx3).
    set2 (numpy array): Second set of coordinates (Mx3).
    num_workers (int): Number of workers for parallel processing.
    
    Returns:
    float: The minimum distance between the two sets of points.
    """
    min_dist = np.inf  # Initialize with a large value

    # Use ProcessPoolExecutor to run the distance computation in parallel
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(compute_min_distances, point1, set2) for point1 in set1]
        
        # Iterate through the results as they complete
        for future in as_completed(futures):
            result = future.result()
            min_dist = min(min_dist, result)
    
    return min_dist

atlas_path = ("/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/atlas/atlas_aleksandra/"
"atlas-release-mouse-barrels-density-mod")

ROI = ["ACA",
    "FRP",
    "MOp",
    "MOs",
    "ORB",
    "PL",
    "RSP",
    "ILA",
    "SSp",
    "AUD",
    "SSs",
    "TEa",
    "VISC",
    "ECT",
    "PERI",
    "GU",
    "PTLp",
    "VIS",
    "AI",
    ]

_, brain_regions, region_map = load_atlas(atlas_path, 'brain_regions', 'hierarchy.json')
atlas_hierarchy = atlas_path + "/hierarchy.json"
voxel_dx = brain_regions.voxel_dimensions[0]
print("Dimension of a voxel in um : ", voxel_dx)
# rg_df = region_map.as_dataframe()
# print(rg_df)
# print(rg_df.columns)
# # get all the voxels indices of each ROI
# print(np.where(brain_regions.raw > 0 ))
# values = voxcell.voxel_data.ValueToIndexVoxels(brain_regions.raw)
# indices = values.value_to_1d_indices(value=614454285)
# print(indices)
# print(len(indices))

# compute the distance between the center of each region to the others
dists_dict = {}
borders_dict = {}
# with h5py.File('/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/inputs/region_masks.h5', 'r') as f:
#     for reg in ROI:
#         if reg != 'MOp':
#             continue
#         dists = []
#         id_reg = list(region_map.find(reg, 'acronym'))[0]
#         dataset = f[str(id_reg)]
#         # convert the dataset to a numpy array
#         voxels = dataset[:]
#         borders_dict[reg] = np.load(reg+'.npy') # get_border_voxels(voxels)
#         # save it to not recompute
#         # np.save(reg+'.npy', borders_dict[reg])
#         # compute the center of the data, in um
#         center = np.mean(voxels, axis=0) #* voxel_dx
#         center = np.array([center])
#         print(center)
#         for reg_other in ROI:
#             print("Computing min distance for target region ", reg_other)
#             if reg_other == "MOp":
#                 dist = 0.
#                 dists.append(dist)
#                 continue
#             id_reg = list(region_map.find(reg_other, 'acronym'))[0]
#             dataset = f[str(id_reg)]
#             voxels_other = dataset[:]
#             borders_dict[reg_other] = np.load(reg_other+'.npy') # get_border_voxels(voxels_other)
#             # save it to not recompute
#             # np.save(reg_other+'.npy', borders_dict[reg_other])
#             # center_other = np.mean(voxels, axis=0)# * voxel_dx
#             # using centers of regions
#             # dist = np.linalg.norm(center - center_other)
#             # using closest points
#             # dist = min_distance_between_sets(center, voxels_other) * voxel_dx
#             # using volume borders
#             dist = min_distance_between_sets(borders_dict[reg], borders_dict[reg_other]) * voxel_dx
#             dists.append(dist)
#         dists_dict[reg] = dists

# # and compile that in a df
# dists_df = pd.DataFrame(dists_dict, index=ROI)
# dists_df = dists_df.T
# print(dists_df)
# dists_df.to_csv("dists_df.csv")
dists_df = pd.read_csv("dists_df.csv", index_col=0)
print(dists_df)
# load the number of connections for each pathway
conns_synth_df = pd.read_csv("/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/axon/"
"axonal-projection/axon_projection/validation/circuit-build/lite_MOp5_final/connectivity_with_parents_axons.csv", index_col=0)
conns_local_df = pd.read_csv("/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/axon/"
"axonal-projection/axon_projection/validation/circuit-build/lite_iso_no_axons_new_atlas/connectivity_with_parents_no_axons.csv", index_col=0)
conns_bio_df = pd.read_csv("/gpfs/bbp.cscs.ch/data/project/proj135/home/petkantc/axon/axonal-projection/axon_projection/validation/circuit-build/lite_iso_bio_axons/connection_counts_for_pathways_vs_bio.csv")
with open(atlas_hierarchy, encoding="utf-8") as f:
    hierarchy_data = json.load(f)
hierarchy = hierarchy_data["msg"][0]
parent_mapping = build_parent_mapping(hierarchy)
conns_bio_df["parent_region"] = conns_bio_df["idx-region_post"].apply(
    lambda x: find_parent_acronym(x, parent_mapping, ROI)
)
conns_bio_df = conns_bio_df.dropna(subset=["parent_region"])
conns_bio_df = conns_bio_df.rename(columns={"0": "connectivity_count"})
conns_bio_df = conns_bio_df[conns_bio_df["connectivity_count"] > 0]
conns_bio_df["type"] = "bio"

def get_distance(row):
    source = row['source_region']
    target = row['target_region']
    return int(dists_df.loc[source, target]) # when using cluster centers

dfs_to_process = [conns_synth_df, conns_local_df, conns_bio_df]
nb_axons = [1695., 1695., 63.]
for i, df in enumerate(dfs_to_process):
    df['source_region'] = 'MOp'
    df.rename(columns={'parent_region': 'target_region'}, inplace=True)
    df['distance'] = df.apply(get_distance, axis=1)
    df['total_in_parent'] = df.groupby(['target_region'])['connectivity_count'].transform('sum')
    df = df[['distance', 'total_in_parent', 'type', 'target_region']]
    # normalize by the number of axons
    df.loc[:, 'total_in_parent'] = df['total_in_parent'] / nb_axons[i]
    dfs_to_process[i] = df
    df.to_csv(f"conn_vs_dist_{str(i)}.csv")

# finally, plot the total_in_parent vs distance for each df
df_to_plot = pd.concat(dfs_to_process)
set_font_size()
fig, ax = plt.subplots(figsize=(5, 5))
set_font_size()
tab_blue_rgb = mcolors.to_rgb("tab:blue")
tab_red_rgb = mcolors.to_rgb("tab:red")
tab_green_rgb = mcolors.to_rgb("tab:green")
palette = {"long_range_axons":tab_red_rgb , "local_axons": tab_green_rgb, "bio": tab_blue_rgb}
sns.lineplot(x='distance', y='total_in_parent', hue='type', data=df_to_plot, palette=palette, markers=True,  # Enable markers (like scatterplot)
    style='type',  # Different styles for different 'type'
    dashes=False,  # Disable dashes to keep solid lines
    linewidth=2,   # Line thickness
    marker='o',    # Marker style
    markersize=10  # Size of markers
    )
# Annotate bars with labels from 'target_region'
for index, row in conns_synth_df.iterrows():
    ax.text(
        x=row['distance'], 
        y=np.log(row['total_in_parent']), 
        s=row['target_region'], 
        ha='center', 
        va='bottom',
        rotation=90,
        fontsize=16,
    )

ax.set(xlabel=r'Distance [$\mu$m]', ylabel='Normalized number of connections')
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_yscale("log")
fig.savefig("conn_vs_dist.pdf")