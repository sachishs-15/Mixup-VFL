mkdir Datasets

REAL_ESTATE="https://drive.google.com/uc?export=download&id=1FMhDnBPeoR2TPx5lnlQLG4mhptBOX67B"
SUPERCONDUCTIVITY="https://drive.google.com/uc?export=download&id=1QcnMcEJdrP3LxDDmt1HvQBqnZAqoj2CN"
YACHT="https://drive.google.com/uc?export=download&id=1FMhDnBPeoR2TPx5lnlQLG4mhptBOX67B"
ENERGY_EFFICIENCY="https://drive.google.com/uc?export=download&id=1k_6wqiPuDcL50_e_L9BQvnFxD1qcQ1t7"
CONCRETE_COMPRESSIVE_STRENGTH="https://drive.google.com/uc?export=download&id=1vD683ABSM2y-GePLytNxFn6VL-OP_8d9"

DATASETS=("REAL_ESTATE" "SUPERCONDUCTIVITY" "YACHT" "ENERGY_EFFICIENCY" "CONCRETE_COMPRESSIVE_STRENGTH")

for DATASET in "${DATASETS[@]}"
do
    echo "Downloading $DATASET"
    curl -L -o Datasets/${DATASET}.csv ${!DATASET}
    echo "Downloaded $DATASET"
done

