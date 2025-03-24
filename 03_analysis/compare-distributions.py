"""
This script takes the equilibrated fraction of each simulation
and does a quick comparison of the Gaussian distributions of each
to determine whether they are substantially different.
"""
import collections
import json
import pathlib

import click
import tqdm
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt

sns.set_context("talk")
mpl.rcParams['font.sans-serif'] = ["muli"]


@click.command()
@click.option(
    "--input-path",
    "-i",
    default="output_sage-2-0-0_1",
    help="The directory containing the equilibration data",
)
@click.option(
    "--output-path",
    "-o",
    default="output_sage-2-0-0_1",
    help="The directory to save the output to",
)
@click.option(
    "--image-path",
    "-im",
    default="images",
    help="The directory to save the images to",
)
def main(
    input_path: str = "output_sage-2-0-0_1",
    output_path: str = "output_sage-2-0-0_1",
    image_path: str = "images",
):
    input_directory = pathlib.Path(input_path)
    df = pd.read_csv(input_directory / "equilibration_data.csv")
    print(f"Loaded {len(df)} equilibration data")

    with open(input_directory / "equilibrated_fraction.json", "r") as f:
        equilibrated_fraction = json.load(f)
    box_to_item = {
        item["box"]: item
        for item in equilibrated_fraction
    }

    # get distributions per Substance
    statistics = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(list)
            )
        )
    )
    equilibrated_data = []
    red_flag_substances = []
    for substance, substance_df in df.groupby("Substance"):
        for box, box_df in substance_df.groupby("Box"):
            for prop, prop_df in box_df.groupby("Property"):
                # assert these are all unique
                ids = ["group_id", "full_id", "Id", "Substance", "Box", "Temperature (K)", "Pressure (Pa)"]
                for id_ in ids:
                    assert len(prop_df[id_].unique()) == 1

                full_id = prop_df["full_id"].iloc[0]
                box = prop_df["Box"].iloc[0]
                values = prop_df["Value"]
                stat = box_to_item[box]

                index = stat["equilibration_index"]
                n_total = stat["n_total_points"]

                values_ = values[index:]

                prop_id = prop_df["Id"].iloc[0]
                group_id = prop_df["group_id"].iloc[0]
                temp = prop_df["Temperature (K)"].iloc[0]
                pressure = prop_df["Pressure (Pa)"].iloc[0]

                statistics[substance][temp][pressure][prop].append({
                    "mean": values_.mean(),
                    "std": values_.std(),
                    "property": prop,
                    "group_id": group_id,
                    "full_id": full_id,
                    "Id": prop_id,
                    "Substance": substance,
                    "Box": box,
                    "Temperature (K)": temp,
                    "Pressure (Pa)": pressure
                })

                equilibrated_data.append(
                    prop_df.iloc[index:]
                )

                if len(values) != n_total:
                    print(f"Warning: {len(values)} != {n_total}: {box}")
                    red_flag_substances.append(substance)

    
    
    output_directory = pathlib.Path(output_path)
    output_directory.mkdir(exist_ok=True, parents=True)

    statistics_file = output_directory / "distribution_statistics.json"
    with open(statistics_file, "w") as f:
        json.dump(statistics, f, indent=2)
    print(f"Saved to {statistics_file}")

    # save equilibrated df
    equilibrated_df = pd.concat(equilibrated_data)
    equilibrated_df.to_csv(
        output_directory / "equilibrated_data.csv", index=False
    )

    # for each substance, compare the distributions
    ood_substances = []
    for substance, temp_dict in statistics.items():
        for temp, pressure_dict in temp_dict.items():
            for pressure, prop_dict in pressure_dict.items():
                for prop, stats in prop_dict.items():
                    bounds = [
                        (
                            stat["mean"],
                            stat["mean"] - stat["std"],
                            stat["mean"] + stat["std"]
                        )
                        for stat in stats
                    ]
                    # check if the bounds overlap
                    overlap = True
                    for bound in bounds:
                        for other_bound in bounds:
                            if bound == other_bound:
                                continue
                            if not (
                                bound[1] < other_bound[2]
                                and bound[2] > other_bound[1]
                            ):
                                overlap = False
                                break
                    if len(bounds) == 1:
                        overlap = True # ignore single distributions

                    if substance in red_flag_substances:
                        overlap = False

                    if not overlap:
                        ood_substances.append({
                            "state": [substance, temp, pressure, prop],
                            "stats": stats
                        })
        
    print(
        f"Found {len(ood_substances)} substances with "
        "non-overlapping distributions"
    )
    output_ood = output_directory / "out_of_distribution.json"
    with open(output_ood, "w") as f:
        json.dump(ood_substances, f, indent=2)
    
    print(f"Saved to {output_ood}")

    # plot ood substances
    image_path = pathlib.Path(image_path)
    image_path.mkdir(exist_ok=True, parents=True)
    for data in tqdm.tqdm(ood_substances, desc="Plotting"):
        state = data["state"]
        substance, temp, pressure, _ = state
        subdf = equilibrated_df[
            (equilibrated_df.Substance == substance)
            & (equilibrated_df["Temperature (K)"] == temp)
            & (equilibrated_df["Pressure (Pa)"] == pressure)
        ]
        g = sns.FacetGrid(
            subdf,
            col="Property",
            hue="Box",
            sharex=False,
            sharey=False,
            margin_titles=True,
            aspect=1.3,
            height=5,
        )
        g.map(sns.kdeplot, "Value")
        g.set_titles(col_template="{col_name}")
        g.figure.suptitle(f"{substance} @ {temp} K, {pressure} Pa")
        plt.tight_layout()

        san_substance = substance.replace("/", "_") # for saving

        plt.savefig(image_path / f"ood_{temp}_{pressure}_{san_substance}.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()

