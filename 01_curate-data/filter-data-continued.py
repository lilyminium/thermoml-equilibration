"""
This applies an initial filter to prune out definitively unwanted data.

This builds off https://github.com/openforcefield/openff-sage/blob/main/data-set-curation/physical-property/optimizations/curate-training-set.py
"""
import time
import pathlib
import click

import pandas as pd
import numpy as np

from openff.evaluator.datasets.curation.components import filtering, selection, thermoml
from openff.evaluator.datasets.curation.components.selection import State, TargetState
from openff.evaluator.datasets.curation.workflow import (
    CurationWorkflow,
    CurationWorkflowSchema,
)

TARGET_STATES = [
    TargetState(
        property_types=[
            ("Density", 1),
        ],
        states=[
            State(
                temperature=298.15,
                pressure=101.325,
                mole_fractions=(1.0,),
            ),
        ],
    ),
    TargetState(
        property_types=[
            ("Density", 2),
            ("EnthalpyOfMixing", 2),
        ],
        states=[
            State(
                temperature=298.15,
                pressure=101.325,
                mole_fractions=(inc, 1-inc),
            )
            for inc in np.arange(0.25, 1, 0.25)
        ],
    ),
]


def curate_data_set(
    input_data_frame,
    n_processes,
) -> pd.DataFrame:

    curation_schema = CurationWorkflowSchema(
        component_schemas=[
            # Select the data points for different compositions.
            selection.SelectDataPointsSchema(target_states=TARGET_STATES),
            # Filter out measurements too similar to each other
            filtering.FilterDuplicatesSchema(
                mole_fraction_precision=2,
            ),
            # Filter out the density of water.
            filtering.FilterBySubstancesSchema(substances_to_exclude=[("O",)]),
        ]
    )

    return CurationWorkflow.apply(input_data_frame, curation_schema, n_processes)


@click.command()
@click.option(
    "--output-file",
    "-o",
    default="intermediate/continued-filtered-thermoml.csv",
    help="The output CSV file to save the filtered data to",
)
@click.option(
    "--input-file",
    "-i",
    default="intermediate/initial-filtered-thermoml.csv",
    help="The CSV file containing existing parsed ThermoML data",
)
@click.option(
    "--n-processes",
    "-np",
    default=1,
    help="The number of processes to use for filtering the data",
)
def main(
    input_file: str = "intermediate/initial-filtered-thermoml.csv",
    output_file: str = "intermediate/continued-filtered-thermoml.csv",
    n_processes: int = 1,
):
    now = time.time()
    print(f"Starting at {time.ctime(now)}")

    thermoml_data_frame = pd.read_csv(input_file, index_col=0)
    print(f"Loding {len(thermoml_data_frame)} data")

    # filter out probably bad data -- e.g. 0.0 enthalpies of mixing and densities exactly
    thermoml_data_frame = thermoml_data_frame[thermoml_data_frame["EnthalpyOfMixing Value (kJ / mol)"] != 0.0]
    thermoml_data_frame = thermoml_data_frame[thermoml_data_frame["Density Value (g / ml)"] != 0.0]

    print(f"Filtered to {len(thermoml_data_frame)} data points after removing bad data")

    training_set_frame = curate_data_set(
        thermoml_data_frame,
        n_processes,
    )
    print(f"Filtered to {len(training_set_frame)} data points")


    output_file = pathlib.Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    training_set_frame.to_csv(output_file)
    print(f"Saved to {output_file}")

    print(f"Finished at {time.ctime(time.time())}")
    print(f"Elapsed time: {time.time() - now} seconds")



if __name__ == "__main__":
    main()