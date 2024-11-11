#!/bin/bash

# superPath='/project/smartlab2021/shebd/FYP2024/slyipae/MODELs/others/4222'
# BERTpath="/project/smartlab2021/shebd/FYP2024/slyipae/MODELs/bert-base-uncased"
# DeBERTapath="/project/smartlab2021/shebd/FYP2024/slyipae/MODELs/DeBERTa-v3-base-mnli-fever-anli"

# cd /home/shebd/5_FYP2024/slyipae/ProgressData/others/COMP4222-Course-Project/ori_code/
# python document_generation.py --use-cuda --use-context

# cd /home/shebd/5_FYP2024/slyipae/ProgressData/others/COMP4222-Course-Project/ori_code/graph

################

superPath='/home/sunanhe/fyp2024slyipae/others/COMP4222-Course-Project/ori_code'
BERTpath="/data/shebd/fyp2024slyipae/MODELs/others/bert-base-uncased"
DeBERTapath="/data/shebd/fyp2024slyipae/MODELs/others/DeBERTa-v3-base-mnli-fever-anli"

cd '/home/sunanhe/fyp2024slyipae/others/COMP4222-Course-Project/ori_code/graph'

############################
# original reproduction

dataName="data"
weightsName="weights"
imageName="images"
# bert
# python make_graph.py --use-cuda --path "${superPath}/${dataName}/" --model_name "${BERTpath}"
# python contrastive_learning.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --save-model
# python train_graph.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --save-model

# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "train"
# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "val"
# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "test"
# python visualize_graph.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${imagesName}/" --weights_folder "${superPath}/${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/"


# DeBERTa
dataName="data_DeBERTa"
weightsName="weights_DeBERTa"
imageName="images_DeBERTa"
# python make_graph_DeBERTa.py --use-cuda --path "${superPath}/${dataName}/" --model_name "${DeBERTapath}"
# python contrastive_learning.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --save-model
# python train_graph.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --save-model

# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "train"
# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "eval"
# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "test"
# python visualize_graph.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${imagesName}/" --weights_folder "${superPath}/${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --save-model

# ############################
# # 0.4 reproduction

# # bert
dataName="data_t0.7"
weightsName="weights_t0.7"
imageName="images_t0.7"

python make_graph.py --use-cuda --path "${superPath}/${dataName}/" --model_name "${BERTpath}" --threshold 0.7
# python contrastive_learning.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --save-model
# python train_graph.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --save-model

# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "train"
# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "eval"
# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "test"
# python visualize_graph.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${imagesName}//" --weights_folder "${superPath}/${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --save-model

# # DeBERTa
# dataName="data_DeBERTa_t0.7"
# weightsName="weights_DeBERTa_t0.7"
# imageName="images_DeBERTa_t0.7"

# python make_graph_DeBERTa.py --use-cuda --path "${superPath}/${dataName}/" --model_name "${DeBERTapath}" --threshold 0.7
# python contrastive_learning.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --save-model
# python train_graph.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --save-model

# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "train"
# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "eval"
# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "test"
# python visualize_graph.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${imagesName}/" --weights_folder "${superPath}/${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --save-model