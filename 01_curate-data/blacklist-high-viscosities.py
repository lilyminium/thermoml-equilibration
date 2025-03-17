"""
Filter the ThermoML dataset for properties that comprise components with viscosities below a certain threshold.

This saves a new CSV file with the filtered properties.
"""
import time

import click
import pathlib
import pandas as pd
import numpy as np
import tqdm

from openff.evaluator.datasets import PhysicalPropertyDataSet


@click.command()
@click.option(
    "--input-file",
    "-i",
    default="intermediate/continued-filtered-thermoml.csv",
    help="The CSV file containing existing parsed ThermoML data",
)
@click.option(
    "--viscosity-file",
    "-v",
    default="viscosities/viscosity-stats.csv",
    help="The CSV file containing viscosity max/min/mean values by SMILES",
)
@click.option(
    "--threshold",
    "-t",
    default=0.3,
    help=(
        "The maximum viscosity value (Pa * s) to filter by. "
        "All output properties will contain components "
        "with max viscosity <= threshold"
    )
)
@click.option(
    "--output-file",
    "-o",
    default="intermediate/continued-filtered-without-high-viscosities.csv",
    help="The directory to save the filtered properties CSV file",
)
def main(
    input_file: str = "intermediate/continued-filtered-thermoml.csv",
    viscosity_file: str = "viscosities/viscosity-stats.csv",
    threshold: float = 0.5,
    output_file: str = "intermediate/continued-filtered-without-high-viscosities.csv",
):
    now = time.time()
    print(f"Starting at {time.ctime(now)}")
    
    # load from previously parsed data to save re-parsing
    df = pd.read_csv(input_file, index_col=0)
    print(f"Loaded {len(df)} properties")

    # load viscosity values
    viscosities = pd.read_csv(viscosity_file, index_col=0)
    assert len(viscosities.smiles.unique()) == len(viscosities)

    print(f"Loaded {len(viscosities)} viscosity values")

    # use median in case of weird outliers
    filtered = viscosities[viscosities["median"] > threshold]
    print(f"Found {len(filtered)} median viscosity values > {threshold}")
    
    filtered_properties = []

    # filter as a dataframe because dataset conversion is quite slow
    smiles_columns = [x for x in df.columns if x.startswith("Component ")]
    for _, row in tqdm.tqdm(df.iterrows(), desc="Filtering", total=len(df)):
        smiles = [row[col] for col in smiles_columns if pd.notna(row[col])]

        if "O" in smiles: # allow water in
            smiles.remove("O")
            
        # now check for viscosities
        if any(smi in filtered.smiles.values for smi in smiles):
            continue

        filtered_properties.append(row)

    filtered_df = pd.DataFrame(filtered_properties)
    print(f"Filtered to {len(filtered_df)} properties")


    output_file = pathlib.Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_csv(output_file)
    print(f"Saved to {output_file}")

    print(f"Finished at {time.ctime(time.time())}")
    print(f"Elapsed time: {time.time() - now} seconds")


if __name__ == "__main__":
    main()
