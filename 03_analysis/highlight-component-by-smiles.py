import collections
import json
import pathlib

import click
import tqdm
import numpy as np
import pandas as pd

from openff.evaluator.datasets.datasets import PhysicalPropertyDataSet, PhysicalProperty
from openff.evaluator.utils.timeseries import TimeSeriesStatistics

import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt

sns.set_context("talk")
mpl.rcParams['font.sans-serif'] = ["muli"]

@click.command()
@click.option(
    "--input-directory",
    "-i",
    default="output",
    help="The directory containing the equilibration data",
)
@click.option(
    "--image-directory",
    "-im",
    default="images",
    help="The directory to save the images to",
)
@click.option(
    "--output-directory",
    "-o",
    default="output",
    help="The directory to save the output to",
)
@click.option(
    "--include-smiles",
    "-ic",
    default="include-molecules.smi",
    help="Components to include"
)
def main(
    input_directory: str = "output",
    image_directory: str = "images",
    output_directory: str = "output",
    include_smiles="include-molecules.smi",
):
    input_directory = pathlib.Path(input_directory)

    input_statistics_file = input_directory / "equilibrated_fraction.json"
    with open(input_statistics_file, "r") as f:
        equilibrated_fraction = json.load(f)

    statistics_by_box = {
        str(item["box"]): item
        for item in equilibrated_fraction
    }
    

    with open(include_smiles, "r") as f:
        include_smi = [x.strip().split()[0] for x in f.readlines()]

    unique_molecules = collections.defaultdict(list)
    for smi in include_smi:
        for item in equilibrated_fraction:
            if smi in item["components"]:
                unique_molecules[smi].append({
                    "property_id": item["property_id"],
                    "equilibrated_fraction": item["equilibrated_fraction"],
                })


    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    # get all properties involving these molecules
    input_equilibration_file = input_directory / "equilibration_data.csv"
    df = pd.read_csv(input_equilibration_file)
    print(f"Loaded {len(df)} equilibration data")

    subset_df = pd.DataFrame(
        df[
            (df["Component 1"].isin(unique_molecules)) |
            (df["Component 2"].isin(unique_molecules))
        ]
    )
    unique_ids = subset_df["Id"].unique()
    n_unique = len(unique_ids)
    print(f"Found {n_unique} properties involving these molecules")


    # split up plotting by smiles
    for smi, properties in unique_molecules.items():
        smi_ids = [x["property_id"] for x in properties]
        subset_df_smi = subset_df[
            (subset_df["Id"].isin(smi_ids))
            # filter out irrelevant single-component contributions
            & ([smi in x for x in subset_df["Substance_"]])
            # also chop off first few frames
            & (subset_df["Time (ns)"] > 0.2)
        ]

        g = sns.FacetGrid(
            subset_df_smi,
            row="Substance_", col="Property",
            hue="full_id",
            sharey=False, sharex=True,
            margin_titles=True,
            aspect=1.2,
            height=5,
        )
        g.map_dataframe(sns.lineplot, x="Time (ns)", y="Value")
        g.set_titles(row_template="{row_name}", col_template="{col_name}")
        g.set_axis_labels("Time (ns)", "Value")

        for (row_name, col_name), ax in g.axes_dict.items():
            subdf_ = subset_df_smi[subset_df_smi["Substance_"] == row_name]
            boxes = subdf_["Box"].unique()
            for box in boxes:
                equilibration_index = statistics_by_box[box]["equilibration_index"]
                equilibration_ns = equilibration_index / 100
                ax.axvline(equilibration_ns, color="red", linestyle="--")

        image_directory = pathlib.Path(image_directory)
        image_directory.mkdir(exist_ok=True, parents=True)
        image_file = image_directory / f"problematic_molecules_{smi}.png"
        plt.tight_layout()
        plt.savefig(image_file, dpi=300)
        print(f"Plotted to {image_file}")


if __name__ == "__main__":
    main()
