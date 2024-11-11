#!/bin/bash

########################################################################################

superPath='/4222'
BERTpath="MODELs/bert-base-uncased"
DeBERTapath="MODELs/DeBERTa-v3-base-mnli-fever-anli"

cd COMP4222-Course-Project/ori_code/graph

############################
# bert 60% 0.85(ori)
resultFolder="$superPath/bert_60per_t0.85"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph.py --use-cuda --path "${dataName}/" --model_name "${BERTpath}" --threshold 0.85 --output_txt "$resultFolder/make_graph.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "train"
# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "val"
python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/"

############################
# bert 60% 0.75
resultFolder="$superPath/bert_60per_t0.75"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph.py --use-cuda --path "${dataName}/" --model_name "${BERTpath}" --threshold 0.75 --output_txt "$resultFolder/make_graph.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "train"
# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "val"
python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/"


############################
# bert 60% 0.95
resultFolder="$superPath/bert_60per_t0.95"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph.py --use-cuda --path "${dataName}/" --model_name "${BERTpath}" --threshold 0.95 --output_txt "$resultFolder/make_graph.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "train"
# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "val"
python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/"


############################
# deberta 60% 0.85(ori)
resultFolder="$superPath/deberta_60per_t0.85"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph_DeBERTa.py --use-cuda --path "${dataName}/" --model_name "${DeBERTapath}" --threshold 0.85 --output_txt "$resultFolder/make_graph_DeBERTa.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "train"
# python evaluate_graph.py --use-cuda --path "${superPath}/${dataName}/" --load-model "${superPath}/${weightsName}/" --mode "val"
python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/"