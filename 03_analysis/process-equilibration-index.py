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


def get_smiles(
    group: str,
    reference_property: PhysicalProperty,
):
    if group == "group": # this is a single-component property
        if len(reference_property.substance.components) > 1:
            smi = str(reference_property.substance)
        else:
            smi = reference_property.substance.components[0].smiles
    elif group == "mixture":
        smi = str(reference_property.substance)
    else:
        index = int(group)
        smi = reference_property.substance.components[index].smiles
    return smi

def _substance_readable(substance):
    smi = " + ".join([
        f"{comp.smiles} ({substance.amounts[comp.identifier][0].value:.2f})"
        for comp in substance.components
    ])
    return smi

def get_readable_substance(
    group: str,
    reference_property: PhysicalProperty,
):
    substance = reference_property.substance
    if group == "group": # this is a single-component property
        if len(reference_property.substance.components) > 1:
            smi = _substance_readable(substance)
        else:
            smi = reference_property.substance.components[0].smiles
    elif group == "mixture":
        smi = _substance_readable(substance)
    else:
        index = int(group)
        smi = reference_property.substance.components[index].smiles
    return smi


def parse_single_box(
    csv_file: pathlib.Path,
    reference_properties_by_id: dict[str, PhysicalProperty]
):
    # parse property attributes
    parent = csv_file.parent.parent
    property_id = parent.name.split("_")[0]
    batch_id = parent.parent.name
    group_id = parent.parent.parent.parent.parent.name
    group = parent.name.split("_")[-1]
    reference_property = reference_properties_by_id[property_id]
    smi = get_smiles(group, reference_property)
    
    # read dataframe
    columns = [
        'Step', 'Potential Energy (kJ/mole)', 'Kinetic Energy (kJ/mole)',
        'Total Energy (kJ/mole)', 'Temperature (K)', 'Box Volume (nm^3)',
        'Density (g/mL)', 'Speed (ns/day)'
    ]
    df = pd.read_csv(csv_file, comment="#", names=columns)
    df["Time (ns)"] = df.Step / 500_000
    melted = df.melt(
        id_vars=["Time (ns)"],
        value_vars=['Potential Energy (kJ/mole)', 'Density (g/mL)'],
        var_name="Property",
        value_name="Value",
    )
    full_id = f"{batch_id}_{property_id}"
    melted["Id"] = property_id
    melted["batch_id"] = batch_id
    melted["group_id"] = group_id
    melted["full_id"] = full_id
    melted["Substance"] = smi
    melted["Substance_"] = get_readable_substance(group, reference_property)
    melted["Box"] = f"{smi} ({full_id})"
    melted["Component 1"] = reference_property.substance.components[0].smiles
    if len(reference_property.substance.components) > 1:
        melted["Component 2"] = reference_property.substance.components[1].smiles
    else:
        melted["Component 2"] = ""
    melted["Temperature (K)"] = reference_property.thermodynamic_state.temperature.m
    melted["Pressure (Pa)"] = reference_property.thermodynamic_state.pressure.m
    return melted


@click.command()
@click.option(
    "--input-directory",
    "-i",
    default="../02_equilibrate-broad",
    help="Directory containing the equilibration data",
)
@click.option(
    "--reference-dataset-path",
    "-r",
    default="../01_curate-data/output/broad-dataset.json",
    help="Path to the reference dataset",
)
@click.option(
    "--output-directory",
    "-o",
    default="output",
    help="Directory to save the output",
)
@click.option(
    "--image-directory",
    "-im",
    default="images",
    help="Directory to save the images",
)
def main(
    input_directory: str = "../02_equilibrate-broad",
    reference_dataset_path: str = "../01_curate-data/output/broad-dataset.json",
    output_directory: str = "output",
    image_directory: str = "images",
):
    reference_dataset = PhysicalPropertyDataSet.from_json(reference_dataset_path)
    reference_properties_by_id = {
        physprop.id: physprop
        for physprop in reference_dataset.properties
    }

    input_directory = pathlib.Path(input_directory)

    working_directory = (
        input_directory
        / "working-directory"
        / "EquilibrationLayer"
    )

    csv_files = sorted(working_directory.glob("*/*/*/openmm_statistics.csv"))
    dfs = []
    for csv_file in tqdm.tqdm(csv_files, desc="Parsing CSVs"):
        try:
            df = parse_single_box(csv_file, reference_properties_by_id)
            dfs.append(df)
        except Exception as e:
            print(f"Error parsing {csv_file}: {e}")
    df = pd.concat(dfs)
    
    # parse jsons
    extract_jsons = sorted(working_directory.glob("*/*/*extract*/*.json"))
    property_statistics = collections.defaultdict(lambda: collections.defaultdict(dict))
    for json_file in tqdm.tqdm(extract_jsons, desc="Parsing JSONs"):
        observable = json_file.name.split("_")[2]
        batch_directory = json_file.parent.parent.parent
        batch_id = batch_directory.name
        group_id = batch_directory.parent.parent.parent.name

        with json_file.open("r") as f:
            contents = json.load(f)
        statistics = TimeSeriesStatistics.parse_json(
            json.dumps(contents[".time_series_statistics"])
        )
        property_id = str(json_file.name.split("|")[0].strip())
        group = json_file.parent.parent.name.split("_")[-1]
        reference_property = reference_properties_by_id[property_id]
        smi = get_smiles(group, reference_property)
        property_statistics[f"{batch_id}_{property_id}"][smi][observable] = {
            "property_id": property_id,
            "batch_id": batch_id,
            "group_id": group_id,
            "statistical_inefficiency": statistics.statistical_inefficiency,
            "equilibration_index": statistics.equilibration_index,
            "n_total_points": statistics.n_total_points,
            "n_uncorrelated_points": statistics.n_uncorrelated_points,
        }

    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    image_directory = pathlib.Path(image_directory)
    image_directory.mkdir(parents=True, exist_ok=True)

    output_statistics_file = output_directory / "property_statistics.json"
    with output_statistics_file.open("w") as f:
        json.dump(property_statistics, f, indent=2)
    print(f"Saved {output_statistics_file}")

    output_df_file = output_directory / "equilibration_data.csv"
    df.to_csv(output_df_file, index=False)
    print(f"Saved {output_df_file}")

    property_names = {
        "Potential Energy (kJ/mole)": "PotentialEnergy",
        "Density (g/mL)": "Density",
    }

    # get fractions of equilibrated data
    equilibrated_fraction = []
    for full_id, smiles_statistics in property_statistics.items():
        batch_id, property_id = full_id.split("_")
        physprop = reference_properties_by_id[property_id]
        for smi, statistics in smiles_statistics.items():
            n_total_points_both = [
                stats["n_total_points"] for stats in statistics.values()
            ]
            equilibration_index_both = [
                stats["equilibration_index"] for stats in statistics.values()
            ]

            # these should be the same for all properties
            assert len(set(n_total_points_both)) == 1
            equilibration_index = max(equilibration_index_both)
            fraction = equilibration_index / n_total_points_both[0]
            equilibrated_fraction.append({
                "property_id": property_id,
                "batch_id": batch_id,
                "full_id": full_id,
                "substance": smi,
                "box": f"{smi} ({full_id})",
                "equilibration_index": equilibration_index,
                "n_total_points": n_total_points_both[0],
                "equilibrated_fraction": fraction,
                "components": sorted(set([
                    comp.smiles for comp in physprop.substance.components
                ]))
            })
    
    equilibrated_fraction.sort(key=lambda x: x["equilibrated_fraction"])
    equilibrated_df = pd.DataFrame(equilibrated_fraction)

    output_csv = output_directory / "equilibrated_fraction.csv"
    equilibrated_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv}")

    output_json = output_directory / "equilibrated_fraction.json"
    with output_json.open("w") as f:
        json.dump(equilibrated_fraction, f, indent=2)
    print(f"Saved {output_json}")

    # print out the worst equilibrated systems
    subset_equilibrated_df = equilibrated_df[
        equilibrated_df.equilibrated_fraction > 0.5
    ]
    print(
        f"{len(subset_equilibrated_df)} systems >0.5:"
    )
    print(subset_equilibrated_df)



if __name__ == "__main__":
    main()

