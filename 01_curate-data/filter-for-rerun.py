"""
This applies an initial filter to prune out definitively unwanted data.

This builds off https://github.com/openforcefield/openff-sage/blob/main/data-set-curation/physical-property/optimizations/curate-training-set.py
"""
import time
import pathlib
import click

import pandas as pd
import numpy as np

from openff.evaluator.datasets.datasets import PhysicalPropertyDataSet
from openff.evaluator.datasets.curation.components import filtering, selection, thermoml
from openff.evaluator.datasets.curation.components.selection import State, TargetState
from openff.evaluator.datasets.curation.workflow import (
    CurationWorkflow,
    CurationWorkflowSchema,
)

from openff.evaluator.utils.checkmol import ChemicalEnvironment

CHEMICAL_ENVIRONMENTS = [
    # "rare" groups -- OCCO, CC(C)O
    ChemicalEnvironment.CarboxylicAcid, # acetic acid
    ChemicalEnvironment.HalogenDeriv, # chloroform
    ChemicalEnvironment.Aqueous, # water
    ChemicalEnvironment.Nitrile,
    ChemicalEnvironment.Acetal, # C1COCO1
    ChemicalEnvironment.Aldehyde,

    # not found but keep it in anyway?
    ChemicalEnvironment.Cyanate,
    ChemicalEnvironment.Isocyanate,

    # these are distinct enough to try to grab both
    ChemicalEnvironment.PrimaryAliphAmine,
    ChemicalEnvironment.PrimaryAromAmine,

    # amines
    ChemicalEnvironment.SecondaryAmine,
    ChemicalEnvironment.TertiaryAmine,

    # halogens
    ChemicalEnvironment.AlkylChloride,
    ChemicalEnvironment.ArylChloride,
    ChemicalEnvironment.AlkylBromide,
    ChemicalEnvironment.ArylBromide,


    ChemicalEnvironment.Alkane,
    ChemicalEnvironment.Alkene,
    ChemicalEnvironment.Alcohol,
    ChemicalEnvironment.Ketone,
    ChemicalEnvironment.CarboxylicAcidEster,
    ChemicalEnvironment.Ether,
    ChemicalEnvironment.Aromatic,

    # amides
    ChemicalEnvironment.CarboxylicAcidPrimaryAmide,
    ChemicalEnvironment.CarboxylicAcidSecondaryAmide,
    ChemicalEnvironment.CarboxylicAcidTertiaryAmide,
    
    ChemicalEnvironment.Heterocycle,

]

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
                mole_fractions=(0.5, 0.5),
            )
        ],
    ),
]


def curate_data_set(
    input_data_frame,
    n_processes,
) -> pd.DataFrame:
    
    allowed_elements = [
        "C", "O", "N", "Cl", "Br", "H",
    ]

    curation_schema = CurationWorkflowSchema(
        component_schemas=[
            # Only errored molecules
            filtering.FilterBySmilesSchema(
                smiles_to_include=[
                    "BrCCBr",
                    "OCCOCCO",
                    "COC=O",
                    "CCCCCCCCO",
                    "CN1CCNCC1",
                ]
            ),
        ]
    )

    return CurationWorkflow.apply(input_data_frame, curation_schema, n_processes)


@click.command()
@click.option(
    "--input-file",
    "-i",
    default="output/broad-dataset.csv",
    help="The output CSV file to save the filtered data to",
)
@click.option(
    "--output-file",
    "-o",
    default="output/broad-dataset-rerun.csv",
    help="The CSV file containing existing parsed ThermoML data",
)
@click.option(
    "--n-processes",
    "-np",
    default=1,
    help="The number of processes to use for filtering the data",
)
def main(
    input_file: str = "output/broad-dataset.csv",
    output_file: str = "output/broad-dataset-rerun.csv",
    n_processes: int = 1,
):
    now = time.time()
    print(f"Starting at {time.ctime(now)}")

    thermoml_data_frame = pd.read_csv(input_file, index_col=0)
    print(f"Loding {len(thermoml_data_frame)} data")

    training_set_frame = curate_data_set(
        thermoml_data_frame,
        n_processes,
    )
    print(f"Filtered to {len(training_set_frame)} data points")


    output_file = pathlib.Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    training_set_frame.to_csv(output_file)
    print(f"Saved to {output_file}")

    ds = PhysicalPropertyDataSet.from_pandas(training_set_frame)
    ds.json(output_file.with_suffix(".json"), format=True)
    print(f"Saved to {output_file.with_suffix('.json')}")

    print(f"Finished at {time.ctime(time.time())}")
    print(f"Elapsed time: {time.time() - now} seconds")



if __name__ == "__main__":
    main()
