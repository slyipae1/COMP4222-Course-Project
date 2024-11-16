#!/bin/bash


superPath='/reslut/qwen/data'

cd /COMP4222-Course-Project/data_preprocess

################

python document_generation_qwen.py --use-cuda --use-context --output_dir $superPath --path "$superPath/sampled_data.json"

python document_generation_qwen.py --use-cuda --output_dir $superPath --path "$superPath/sampled_data.json"


################

superPath='/reslut/qwen_result'
BERTpath="google-bert/bert-base-uncased"

cd COMP4222-Course-Project/pipeline/graph

resultFolder="$superPath/qwen_result"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph.py --use-cuda --path "${dataName}/" --model_name "${BERTpath}" --threshold ??? --output_txt "$resultFolder/make_graph.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/GAT_???.pt" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --output_txt "$resultFolder/kNN.txt"
