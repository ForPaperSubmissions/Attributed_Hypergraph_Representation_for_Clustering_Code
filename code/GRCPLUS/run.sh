

for dataset in coau_cora coci_cora
do
    python train.py \
        --dataset $dataset
done