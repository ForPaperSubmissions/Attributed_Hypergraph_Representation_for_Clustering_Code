#!/bin/sh

# for dataset in coau_cora coci_citeseer coci_pubmed coci_cora 20news coci_patent_C13 coci_wiki zoo Mushroom NTU2012 ModelNet40 coau_dblp
# for drop_dyadic_rate in 0.1 0.3 0.5 0.7 0.9
# for w_d in 0.0625 0.125 0.25 0.5 1 2 4 8 16

for dataset in coau_dblp
do
    python train.py \
        --dataset $dataset
done


# for dataset in zoo Mushroom NTU2012 ModelNet40
# do
#     for drop_edge_rate_1 in 0.0 0.1 0.2 0.3 0.4
#     do
#         for drop_feature_rate_1 in 0.0 0.1 0.2 0.3 0.4
#         do
#             python train.py \
#                 --dataset $dataset \
#                 --num_seeds 5 \
#                 --drop_edge_rate_1 $drop_edge_rate_1 \
#                 --drop_feature_rate_1 $drop_feature_rate_1
#         done
#     done
# done

# for dataset in 20news
# do
#     for learning_rate in 1.0e-5 5.0e-5 1.0e-4 5.0e-4 1.0e-3 5.0e-3
#     do
#         python train.py \
#             --dataset $dataset \
#             --num_seeds 5 \
#             --learning_rate $learning_rate
#     done
# done

# for dataset in coci_pubmed 20news coau_dblp
# do
#     for drop_edge_rate_1 in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#     do
#         for drop_feature_rate_1 in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#         do
#             python train.py \
#                 --dataset $dataset \
#                 --num_seeds 5 \
#                 --drop_edge_rate_1 $drop_edge_rate_1 \
#                 --drop_feature_rate_1 $drop_feature_rate_1
#         done
#     done
# done
