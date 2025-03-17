"""
This applies an initial filter to prune out definitively unwanted data.

This builds off https://github.com/openforcefield/openff-sage/blob/main/data-set-curation/physical-property/optimizations/curate-training-set.py
"""
import pathlib
import time
import click

import pandas as pd

from openff.evaluator.datasets.curation.components import filtering
from openff.evaluator.datasets.curation.workflow import (
    CurationWorkflow,
    CurationWorkflowSchema,
)

def curate_data_set(
    input_data_frame,
    property_type_filter: filtering.FilterByPropertyTypesSchema,
    n_processes,
) -> pd.DataFrame:
    
    allowed_elements = [
        "C", "O", "N", "Cl", "Br", "H",
        "F", "S", "P", "I"
    ]
    curation_schema = CurationWorkflowSchema(
        component_schemas=[
            # Filter out any measurements made for systems with more than
            # two components
            filtering.FilterByNComponentsSchema(n_components=[1, 2]),
            # Remove any duplicate data.
            filtering.FilterDuplicatesSchema(),
            # Filter out data points measured away from ambient conditions.
            filtering.FilterByTemperatureSchema(
                minimum_temperature=288.15, maximum_temperature=318.15
            ),
            filtering.FilterByPressureSchema(
                minimum_pressure=99.9, maximum_pressure=101.4
            ),
            # Retain only density and enthalpy of mixing data points which
            # have been measured for the same systems.
            property_type_filter,
            # Filter out long chain molecules (slower to simulate / converge) and 1, 3
            # carbonyl compounds where one of the carbonyls is a ketone (cases where
            # the enol form may be present in non-negligible amounts).
            filtering.FilterBySmirksSchema(
                smirks_to_exclude=[
                    # Long chain alkane /ether
                    "-".join(["[#6X4,#8X2]"] * 10),
                    # 1, 3 carbonyls with at least one ketone carbonyl.
                    "[#6](=[#8])-[#6](-[#1])(-[#1])-[#6](=[#8])-[#6]",
                ],
            ),
            # Filter out problematic molecules
            filtering.FilterBySmilesSchema(
                smiles_to_exclude=[
                    # Heavy water.
                    "[2H]O[2H]",
                    # Molecules which OpenMM misinterprets
                    "N[C@@H](CS)C(=O)O",
                    "CSCC[C@H](N)C(=O)O",
                    # Molecules which cause NaNs during simulations
                    "O=S(=O)(O)CCCN1CCOCC1",
                ]
            ),
            # Filter out systems where one component is in a significant excess.
            filtering.FilterByMoleFractionSchema(
                mole_fraction_ranges={2: [[(0.05, 0.95)]]}
            ),
            # Filter out any racemic mixtures
            filtering.FilterByRacemicSchema(),
            # Remove any substances measured for systems with undefined
            # stereochemistry
            filtering.FilterByStereochemistrySchema(),
            # Remove any measurements made for systems where any of the components
            # are charged.
            filtering.FilterByChargedSchema(),
            # Remove measurements made for ionic liquids
            filtering.FilterByIonicLiquidSchema(),
            # Remove any molecules containing elements that aren't currently of interest
            filtering.FilterByElementsSchema(allowed_elements=allowed_elements),
        ]
    )

    return CurationWorkflow.apply(input_data_frame, curation_schema, n_processes)


@click.command()
@click.option(
    "--input-file",
    "-i",
    default="input/thermoml.csv",
    help="The CSV file containing existing parsed ThermoML data",
)
@click.option(
    "--output-file",
    "-o",
    default="intermediate/initial-filtered-thermoml.csv",
    help="The CSV file to save the filtered properties to",
)
@click.option(
    "--n-processes",
    "-np",
    default=1,
    help="The number of processes to use for filtering the data",
)
def main(
    input_file: str = "input/thermoml.csv",
    output_file: str = "intermediate/initial-filtered-thermoml.csv",
    n_processes: int = 1,
):
    now = time.time()
    print(f"Starting at {time.ctime(now)}")

    thermoml_data_frame = pd.read_csv(input_file, index_col=0)
    print(f"Loading {len(thermoml_data_frame)} data")

    training_set_frame = curate_data_set(
        thermoml_data_frame,
        filtering.FilterByPropertyTypesSchema(
            property_types=[
                "Density",
                "EnthalpyOfMixing",
            ],
            n_components={
                "Density": [1, 2],
                "EnthalpyOfMixing": [2],
            },
            strict=True,
        ),
        n_processes,
    )
    print(f"Filtered to {len(training_set_frame)} data points")

    output_file = pathlib.Path(output_file)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    training_set_frame.to_csv(output_file)
    print(f"Saved to {output_file}")

    print(f"Finished at {time.ctime(time.time())}")
    print(f"Elapsed time: {time.time() - now} seconds")



if __name__ == "__main__":
    main()