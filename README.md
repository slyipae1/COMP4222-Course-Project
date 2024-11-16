# Understanding Hallucinations in LLMs: A Graph-Based Reproduction Study (COMP4222 Course project)



## Overview
We performed experiments on the pipeline of the paper ["Leveraging Graph Structures to Detect Hallucinations in Large Language Models"](https://github.com/noanonkes/Hallucination-Detection-in-LLMs) regarding similarity threshold, embedding model, LLM, and dataset.

## Group Members & Task Distibution:
+ YIP Sau Lai: similarity threshold & embedding model
    + Replaced BERT with DeBERTa: pipeline/graph/make_graph_DeBERTa.py
    + For each embedding model, used thresholds: 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
    + Result: result/threshold&embed.tar.gz

+ LAM Sum Ying: LLM for generating hallucination data
    + replaced the original LLM (LLama2) with Qwen2.5-14B to generate data with the original dataset: data_preprocess/document_generation_qwen.py
    + result: result/qwen_result

+ Peng Muzi: dataset for generating hallucination data
    + replaced the original dataset with a new dataset, SciQ with the original LLM: data_preprocess/reformat_SciQ.py
    + result: result/SciQ_result

## Reproduction Guidance
+ similarity threshold & embedding model
    + Create result folders with corresponding names under: result/threshold&embed/
    + Copy "data" folder from pipeline/data, create "weights" and "images" folder in each result folder.
    + Run script: jobScript/threshold&embed.s

+ Qwen as LLM
    + Copy sample_data.json from pipeline/data/sampled_data.json to result/qwen_result
    + Run script: jobScript/qwen_result.s (first part)
    + Create "weights" and "images" folder under result/qwen_result
    + Run script: jobScript/qwen_result.s (second part)

+ SciQ as dataset
    + Run file: data_preprocess/reformat_SciQ.py
    + Run script: jobScript/SciQ_result.s (first part)
    + Create "weights" and "images" folder under result/SciQ_result
    + Run script: jobScript/SciQ_result.s (second part)


## References

[1] N. Nonkes, S. Agaronian, E. Kanoulas, and R. Petcu, "Leveraging Graph Structures to Detect Hallucinations in Large Language Models," in Proceedings of TextGraphs-17: Graph-based Methods for Natural Language Processing, Bangkok, Thailand, Aug. 2024, pp. https://aclanthology.org/2024.textgraphs-1.7

[2] Johannes Welbl, Nelson F. Liu, Matt Gardner, SciQ: "Crowdsourcing Multiple Choice Science Questions", proceedings of the Workshop on Noisy User-generated Text (W-NUT) 2017.

