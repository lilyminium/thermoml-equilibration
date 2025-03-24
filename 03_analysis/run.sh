#!/bin/bash

for FFDIR in sage-2-0-0_1 sage-2-2-1_1 ; do

    OUTPUT_DIR="output_${FFDIR}"
    IMAGE_DIR="images_${FFDIR}"

    # python process-equilibration-index.py \
    #     -i "../02_equilibrate-broad/${FFDIR}" \
    #     -im $IMAGE_DIR \
    #     -o $OUTPUT_DIR > "logs/process-equilibration-index-${FFDIR}.log"

    # python highlight-bad-components.py -t 0.7 \
    #     -i $OUTPUT_DIR \
    #     -im $IMAGE_DIR \
    #     -o $OUTPUT_DIR > "logs/highlight-bad-components-${FFDIR}.log"

    python highlight-component-by-smiles.py \
        -ic include-molecules.smi \
        -i $OUTPUT_DIR \
        -im $IMAGE_DIR \
        -o $OUTPUT_DIR > "logs/highlight-component-by-smiles-${FFDIR}.log"

    # python compare-distributions.py \
    #     -i $OUTPUT_DIR  \
    #     -o $OUTPUT_DIR  \
    #     -im $IMAGE_DIR > "logs/compare-distributions-${FFDIR}.log"

done