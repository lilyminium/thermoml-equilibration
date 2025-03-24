"""
This applies an initial filter to prune out definitively unwanted data.

This builds off https://github.com/openforcefield/openff-sage/blob/main/data-set-curation/physical-property/optimizations/curate-training-set.py
"""
import time
import pathlib
import click

from openff.evaluator.datasets.datasets import PhysicalPropertyDataSet


def save_dataset(dataset, output_file: pathlib.Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_pandas().to_csv(output_file)
    dataset.json(output_file.with_suffix(".json"), format=True)

    print(f"Saved to {output_file}")
    print(f"Saved to {output_file.with_suffix('.json')}")


@click.command()
@click.argument(
    "input_files",
    nargs=-1,
)
@click.option(
    "--output-file",
    "-o",
    default="output/combined-set.csv",
    help="The output CSV file to save the filtered data to",
)
def main(
    input_files,
    output_file: str = "output/combined-set.csv"
):
    now = time.time()
    print(f"Starting at {time.ctime(now)}")

    properties = []
    for input_file in input_files:
        print(f"Reading {input_file}")
        ds = PhysicalPropertyDataSet.from_json(pathlib.Path(input_file))
        properties.extend(ds.properties)
    properties.sort(key=lambda x: x.id)

    ds = PhysicalPropertyDataSet()
    ds.add_properties(*properties)

    save_dataset(ds, pathlib.Path(output_file))
    print(f"Combined to {len(ds.properties)} data points")

    print(f"Finished at {time.ctime(time.time())}")
    print(f"Elapsed time: {time.time() - now} seconds")


if __name__ == "__main__":
    main()
