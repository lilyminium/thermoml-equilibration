import json
import time
import pathlib
import click

import pandas as pd

from openff.evaluator.datasets.datasets import PhysicalPropertyDataSet


@click.command()
@click.option(
    "--input-file",
    "-i",
    default="../01_curate-data/intermediate/continued-filtered-without-high-viscosities.csv",
    help="The input file",
)
@click.option(
    "--output-file",
    "-o",
    default="input/renumbered-dataset.csv",
    help="The output file",
)
def main(
    input_file: str = "../01_curate-data/intermediate/continued-filtered-without-high-viscosities.csv",
    output_file: str = "input/renumbered-dataset.csv"
):
    now = time.time()
    print(f"Starting at {time.ctime(now)}")

    thermoml_data_frame = pd.read_csv(input_file, index_col=0)
    print(f"Loading {len(thermoml_data_frame)} data")

    # renumber the properties
    ds = PhysicalPropertyDataSet.from_pandas(thermoml_data_frame)
    mappings = {
        physprop.id: f"{i:04d}"
        for i, physprop in enumerate(ds.properties, 1)
    }
    with open("mappings.json", "w") as f:
        json.dump(mappings, f, indent=4)

    for prop in ds.properties:
        prop.id = mappings[prop.id]

    output_file = pathlib.Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ds.to_pandas().to_csv(output_file)
    print(f"Renumbered data set saved to {output_file}")

    ds.json(output_file.with_suffix(".json"), format=True)
    print(f"Saved to {output_file.with_suffix('.json')}")

    print(f"Finished at {time.ctime(time.time())}")
    print(f"Elapsed time: {time.time() - now} seconds")


if __name__ == "__main__":
    main()
