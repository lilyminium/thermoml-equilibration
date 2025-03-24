"""
This applies an initial filter to prune out definitively unwanted data.

This builds off https://github.com/openforcefield/openff-sage/blob/main/data-set-curation/physical-property/optimizations/curate-training-set.py
"""
import json
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

    # "rare" groups -- OCCO, CC(C)O
    ChemicalEnvironment.CarboxylicAcid, # acetic acid
    ChemicalEnvironment.HalogenDeriv, # chloroform
    ChemicalEnvironment.Aqueous, # water
    ChemicalEnvironment.Nitrile,
    ChemicalEnvironment.Acetal, # C1COCO1
    ChemicalEnvironment.Aldehyde,

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
                mole_fractions=(0.25, 0.75),
            ),
            State(
                temperature=298.15,
                pressure=101.325,
                mole_fractions=(0.5, 0.5),
            ),
            State(
                temperature=298.15,
                pressure=101.325,
                mole_fractions=(0.75, 0.25),
            )
        ],
    ),
]


def curate_data_set(
    input_data_frame,
    smiles_to_exclude,
    n_processes,
) -> pd.DataFrame:
    
    allowed_elements = [
        "C", "O", "N", "Cl", "Br", "H",
    ]

    curation_schema = CurationWorkflowSchema(
        component_schemas=[
            # Remove any molecules containing elements that aren't currently of interest
            filtering.FilterByElementsSchema(allowed_elements=allowed_elements),
            selection.SelectDataPointsSchema(target_states=TARGET_STATES),
            filtering.FilterBySmilesSchema(
                smiles_to_exclude=smiles_to_exclude,
            ),
        ]
    )

    return CurationWorkflow.apply(input_data_frame, curation_schema, n_processes)


def save_dataset(dataset, output_file: pathlib.Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_pandas().to_csv(output_file)
    dataset.json(output_file.with_suffix(".json"), format=True)

    print(f"Saved to {output_file}")
    print(f"Saved to {output_file.with_suffix('.json')}")

@click.command()
@click.option(
    "--output-file",
    "-o",
    default="output/validation-set.csv",
    help="The output CSV file to save the filtered data to",
)
@click.option(
    "--input-file",
    "-i",
    default="input/renumbered-dataset.json",
    help="The JSON file containing existing parsed ThermoML data",
)
@click.option(
    "--exclude-file",
    "-x",
    default="../03_analysis/output/exclude-molecules.smi",
    help="The file containing SMILES to exclude",
)
@click.option(
    "--n-processes",
    "-np",
    default=1,
    help="The number of processes to use for filtering the data",
)
@click.option(
    "--training-file",
    "-t",
    default="output/training-set.json",
    help="The JSON file containing the training set",
)
def main(
    input_file: str = "input/renumbered-dataset.json",
    training_file: str = "output/training-set.json",
    exclude_file: str = "../03_analysis/output/exclude-molecules.smi",
    output_file: str = "output/validation-set.csv",
    n_processes: int = 1,
):
    now = time.time()
    print(f"Starting at {time.ctime(now)}")

    ds = PhysicalPropertyDataSet.from_json(pathlib.Path(input_file))
    training_set = PhysicalPropertyDataSet.from_json(pathlib.Path(training_file))

    # filter out training properties
    training_ids = [x.id for x in training_set.properties]
    ds2 = PhysicalPropertyDataSet()
    for prop in ds.properties:
        if prop.id not in training_ids:
            ds2.add_properties(prop)

    thermoml_data_frame = ds2.to_pandas()
    print(f"Loading {len(thermoml_data_frame)} data")

    # load smiles to exclude
    with open(exclude_file, "r") as f:
        contents = f.readlines()
    smiles_to_exclude = [x.strip().split()[0] for x in contents]

    training_set_frame = curate_data_set(
        thermoml_data_frame,
        smiles_to_exclude,
        n_processes,
    )
    print(f"Filtered to {len(training_set_frame)} data points")

    ds = PhysicalPropertyDataSet.from_pandas(training_set_frame)

    save_dataset(ds, pathlib.Path(output_file))

    print(f"Finished at {time.ctime(time.time())}")
    print(f"Elapsed time: {time.time() - now} seconds")



if __name__ == "__main__":
    main()
