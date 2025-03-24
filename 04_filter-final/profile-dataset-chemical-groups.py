import collections
import itertools
import json
import pathlib
import time

import click
import tqdm

import pandas as pd
import numpy as np
from openff.evaluator.utils.checkmol import analyse_functional_groups

@click.command()
@click.option(
    "--input-file",
    "-i",
    default="intermediate/continued-filtered-without-high-viscosities.csv",
    help="The CSV file containing existing parsed ThermoML data",
)
@click.option(
    "--output-directory",
    "-o",
    default="chemical-groups",
    help="The directory to save the filtered properties CSV file",
)
def main(
    input_file: str = "intermediate/continued-filtered-without-high-viscosities.csv",
    output_directory: str = "chemical-groups",
):
    now = time.time()
    print(f"Starting at {time.ctime(now)}")

    df = pd.read_csv(input_file, index_col=0)
    print(f"Loaded {len(df)} properties")

    # get unique SMILES
    all_smiles = set()
    component_cols = [col for col in df.columns if col.startswith("Component ")]
    for col in component_cols:
        all_smiles |= set(df[col].unique())
    all_smiles -= {"", np.nan}

    
    # get chemical groups
    groups = {}
    for smi in tqdm.tqdm(all_smiles):
        groups[smi] = sorted([
            group.value
            for group in analyse_functional_groups(smi)
        ])

    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    groups_file = output_directory / "components.json"
    with open(groups_file, "w") as f:
        json.dump(groups, f, indent=4)
    print(f"Saved groups to {groups_file}")

    # now get mixture groups
    mixture_groups = collections.defaultdict(list)
    
    for _, row in df.iterrows():
        property_groups = []
        for col in component_cols:
            if pd.isna(row[col]):
                continue
            property_groups.append(groups[row[col]])
        for combination in itertools.product(*property_groups):
            key = tuple(sorted(combination))
            mixture_groups[key].append(row["Id"])

    # make json friendly
    mixture_group_list = []
    for combo, property_ids in mixture_groups.items():
        mixture_group_list.append({"groups": list(combo), "property_ids": property_ids})
    
    mixture_output_file = output_directory / "mixture-groups.json"
    with open(mixture_output_file, "w") as f:
        json.dump(mixture_group_list, f, indent=4)
    print(f"Saved mixture groups to {mixture_output_file}")

    # compute counts...
    single_counts = {}
    mixture_counts = {}
    for combo, property_ids in mixture_groups.items():
        counts = len(property_ids)
        if len(combo) == 1:
            single_counts[combo[0]] = counts
        else:
            mixture_counts[combo] = counts

    # make json friendly data formats
    single_count_list = []
    for smiles, count in single_counts.items():
        single_count_list.append({"groups": smiles, "count": count})
    mixture_count_list = []
    for smiles, count in mixture_counts.items():
        mixture_count_list.append({"groups": list(smiles), "count": count})

    # sort
    single_count_list = sorted(single_count_list, key=lambda x: x["count"], reverse=True)
    mixture_count_list = sorted(mixture_count_list, key=lambda x: x["count"], reverse=True)

    print(f"Single component environments: {len(single_count_list)}")
    print(f"Mixture component environments: {len(mixture_count_list)}")

    
    single_counts_file = output_directory / "single-counts.json"
    with open(single_counts_file, "w") as f:
        json.dump(single_count_list, f, indent=4)
    
    mixture_counts_file = output_directory / "mixture-counts.json"
    with open(mixture_counts_file, "w") as f:
        json.dump(mixture_count_list, f, indent=4)


    print(f"Finished at {time.ctime(time.time())}")
    print(f"Elapsed time: {time.time() - now} seconds")


if __name__ == "__main__":
    main()
