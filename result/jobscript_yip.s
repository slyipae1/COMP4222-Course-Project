#!/bin/bash

########################################################################################

superPath='/4222'
BERTpath="MODELs/bert-base-uncased"
DeBERTapath="MODELs/DeBERTa-v3-base-mnli-fever-anli"

cd COMP4222-Course-Project/ori_code/graph

############################
# bert 60% 0.60
resultFolder="$superPath/bert_60per_t0.60"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph.py --use-cuda --path "${dataName}/" --model_name "${BERTpath}" --threshold 0.60 --output_txt "$resultFolder/make_graph.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/GAT_252.pt" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --output_txt "$resultFolder/kNN.txt"


############################
# bert 60% 0.65
resultFolder="$superPath/bert_60per_t0.65"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph.py --use-cuda --path "${dataName}/" --model_name "${BERTpath}" --threshold 0.65 --output_txt "$resultFolder/make_graph.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/GAT_264.pt" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --output_txt "$resultFolder/kNN.txt"



############################
# bert 60% 0.70
resultFolder="$superPath/bert_60per_t0.70"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph.py --use-cuda --path "${dataName}/" --model_name "${BERTpath}" --threshold 0.70 --output_txt "$resultFolder/make_graph.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/GAT_347.pt" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --output_txt "$resultFolder/kNN.txt"


############################
# bert 60% 0.75
resultFolder="$superPath/bert_60per_t0.75"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph.py --use-cuda --path "${dataName}/" --model_name "${BERTpath}" --threshold 0.75 --output_txt "$resultFolder/make_graph.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/GAT_241.pt" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --output_txt "$resultFolder/kNN.txt"

############################
# bert 60% 0.80
resultFolder="$superPath/bert_60per_t0.80"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph.py --use-cuda --path "${dataName}/" --model_name "${BERTpath}" --threshold 0.80 --output_txt "$resultFolder/make_graph.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/GAT_471.pt" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --output_txt "$resultFolder/kNN.txt"


############################
# bert 60% 0.85(ori)
resultFolder="$superPath/bert_60per_t0.85"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph.py --use-cuda --path "${dataName}/" --model_name "${BERTpath}" --threshold 0.85 --output_txt "$resultFolder/make_graph.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/GAT_395.pt" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --output_txt "$resultFolder/kNN.txt"

############################
# bert 60% 0.90
resultFolder="$superPath/bert_60per_t0.90"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph.py --use-cuda --path "${dataName}/" --model_name "${BERTpath}" --threshold 0.90 --output_txt "$resultFolder/make_graph.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/GAT_119.pt" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --output_txt "$resultFolder/kNN.txt"


############################
# bert 60% 0.95
resultFolder="$superPath/bert_60per_t0.95"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph.py --use-cuda --path "${dataName}/" --model_name "${BERTpath}" --threshold 0.95 --output_txt "$resultFolder/make_graph.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/GAT_255.pt" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --output_txt "$resultFolder/kNN.txt"



###########################################################################################################################################################
###########################################################################################################################################################



############################
# DeBERTa 60% 0.60
resultFolder="$superPath/deberta_60per_t0.60"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph_DeBERTa.py --use-cuda --path "${dataName}/" --model_name "${DeBERTapath}" --threshold 0.60 --output_txt "$resultFolder/make_graph_DeBERTa.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/GAT_291.pt" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --output_txt "$resultFolder/kNN.txt"


############################
# DeBERTa 60% 0.65
resultFolder="$superPath/deberta_60per_t0.65"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph_DeBERTa.py --use-cuda --path "${dataName}/" --model_name "${DeBERTapath}" --threshold 0.65 --output_txt "$resultFolder/make_graph_DeBERTa.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/GAT_262.pt" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --output_txt "$resultFolder/kNN.txt"


############################
# DeBERTa 60% 0.70
resultFolder="$superPath/deberta_60per_t0.70"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph_DeBERTa.py --use-cuda --path "${dataName}/" --model_name "${DeBERTapath}" --threshold 0.70 --output_txt "$resultFolder/make_graph_DeBERTa.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/GAT_199.pt" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --output_txt "$resultFolder/kNN.txt"


############################
# DeBERTa 60% 0.75
resultFolder="$superPath/deberta_60per_t0.75"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph_DeBERTa.py --use-cuda --path "${dataName}/" --model_name "${DeBERTapath}" --threshold 0.75 --output_txt "$resultFolder/make_graph_DeBERTa.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/GAT_375.pt" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --output_txt "$resultFolder/kNN.txt"


############################
# DeBERTa 60% 0.80
resultFolder="$superPath/deberta_60per_t0.80"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph_DeBERTa.py --use-cuda --path "${dataName}/" --model_name "${DeBERTapath}" --threshold 0.80 --output_txt "$resultFolder/make_graph_DeBERTa.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/GAT_486.pt" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --output_txt "$resultFolder/kNN.txt"


############################
# deberta 60% 0.85(ori)
resultFolder="$superPath/deberta_60per_t0.85"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph_DeBERTa.py --use-cuda --path "${dataName}/" --model_name "${DeBERTapath}" --threshold 0.85 --output_txt "$resultFolder/make_graph_DeBERTa.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/GAT_273.pt" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --output_txt "$resultFolder/kNN.txt"


############################
# DeBERTa 60% 0.90
resultFolder="$superPath/deberta_60per_t0.90"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph_DeBERTa.py --use-cuda --path "${dataName}/" --model_name "${DeBERTapath}" --threshold 0.90 --output_txt "$resultFolder/make_graph_DeBERTa.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/GAT_449.pt" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --output_txt "$resultFolder/kNN.txt"


############################
# DeBERTa 60% 0.95
resultFolder="$superPath/deberta_60per_t0.95"
dataName="$resultFolder/data"
weightsName="$resultFolder/weights"
imageName="$resultFolder/images"

python make_graph_DeBERTa.py --use-cuda --path "${dataName}/" --model_name "${DeBERTapath}" --threshold 0.95 --output_txt "$resultFolder/make_graph_DeBERTa.txt"
python contrastive_learning.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}" --save-model --output_txt "$resultFolder/contrastive_learning.txt"
python train_graph.py --use-cuda --path "${dataName}/" --output_dir "${weightsName}/" --save-model --output_txt "$resultFolder/train_graph.txt"

python evaluate_graph.py --use-cuda --path "${dataName}/" --load-model "${weightsName}/GAT_391.pt" --mode "test" --output_txt "$resultFolder/evaluate_graph.txt"
python visualize_graph.py --use-cuda --path "${dataName}/" --output_dir "${imageName}/" --weights_folder "${weightsName}/"
# python kNN.py --use-cuda --path "${superPath}/${dataName}/" --output_dir "${superPath}/${weightsName}/" --output_txt "$resultFolder/kNN.txt"

