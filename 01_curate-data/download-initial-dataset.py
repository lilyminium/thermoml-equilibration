"""
Obtain experimental viscosity values from ThermoML.

This script reads ThermoML XML files and extracts viscosity data.
It saves a number of files:

* `thermoml-with-viscosities.csv`: The full dataset with all properties supported by Evaluator, and the viscosity data.
* `viscosities.csv`: The subset of the dataset with only viscosity data.
* `viscosities-subset.csv`: A subset of *single-component* viscosity data in a specific temperature and pressure range (295-305 K, 100-102 kPa).
* `viscosity-stats.csv`: A summary of the viscosity data, including max, min, mean, standard deviation, and count values of each SMILES
* `viscosity-stats.json`: The same summary as `viscosity-stats.csv`, but in JSON format.
"""

import pathlib
import time

import click
import tqdm
import pandas as pd

from openff.evaluator.plugins import register_default_plugins
from openff.evaluator.datasets.curation.components import thermoml

register_default_plugins()


@click.command
@click.option(
    "--output-directory",
    "-o",
    default="input",
    help="The directory to save the output CSV files",
)
@click.option(
    "--n-processes",
    "-np",
    default=1,
    help="The number of processes to use for loading the data",
)
def main(
    output_directory: str = "viscosities",
    n_processes: int = 1,
):
    now = time.time()
    print(f"Starting at {time.ctime(now)}")

    output_directory = pathlib.Path(output_directory)
    output_directory.mkdir(exist_ok=True, parents=True)

    df = thermoml.ImportThermoMLData.apply(
        pd.DataFrame(),
        thermoml.ImportThermoMLDataSchema(
            cache_file_name="input/initial-thermoml.csv"
        ),
        n_processes
    )

    print(f"Downloaded {len(df)} initial data points")

    # filter for mis-formed/multi-component SMILES
    filtered_properties = []

    smiles_columns = [x for x in df.columns if x.startswith("Component ")]
    for _, row in tqdm.tqdm(df.iterrows(), desc="Filtering", total=len(df)):
        smiles = [row[col] for col in smiles_columns if pd.notna(row[col])]
        # filter for multi-component smiles -- not caught in first step
        if any("." in smi for smi in smiles):
            continue

        filtered_properties.append(row)

    df = pd.DataFrame(filtered_properties)
    
    output_path = output_directory / "thermoml.csv"
    df.to_csv(output_path)
    print(
        f"Loaded {len(df)} data points "
        f"and saved to {output_path}"
    )
    print(f"Finished at {time.ctime(time.time())}")
    print(f"Elapsed time: {time.time() - now} seconds")


if __name__ == "__main__":
    main()
