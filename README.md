# Understanding Hallucinations in LLMs: A Graph-Based Reproduction Study (COMP4222 Course project)



## Overview
+ We performed experiments on the pipeline of paper "Leveraging Graph Structures to Detect Hallucinations in Large Language Models" regarding similarity threshold, embedding model, LLM to generate hallucination data, and dataset.

## group members & Task distibution:
+ YIP Sau Lai: conduct enperiments on similarity threshold and embedding model
    + replace BERT with DeBERTa: pipeline/graph/make_graph_DeBERTa.py
    + for each embedding model, used thresholds: 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95
    + result: result/threshold&embed.tar.gz

+ LAM Sum Ying: conduct enperiments on the LLM for generating hallucination data
    + replace the original LLM (LLama2) with Qwen2.5-14B to generate data with the original dataset: data_preprocess/document_generation_qwen.py
    + result: result/qwen_result

+ Peng Muzi: conduct enperiments on the dataset for generating hallucination data
    + replace the original dataset with 1200 sampled from a new dataset, SciQ: data_preprocess/reformat_SciQ.py
    + result: result/SciQ_result

## Reproduction Guidance
+ threshold&embed
    + create result folders with corresponding name under: result/threshold&embed, copy "data" folder from pipeline/data to each result folder, and create "weights" and "images" folder in each result folder.
    + run script: jobScript/threshold&embed.s

+ Qwen as LLM
    + copy sample_data.json from pipeline/data/sampled_data.json to result/qwen_result
    + run script: jobScript/qwen_result.s (first part)
    + create "weights" and "images" folder under result/qwen_result
    + run script: jobScript/qwen_result.s (second part)

+ SciQ as dataset
    + run file: data_preprocess/reformat_SciQ.py
    + run script: jobScript/SciQ_result.s (first part)
    + create "weights" and "images" folder under result/SciQ_result
    + run script: jobScript/SciQ_result.s (second part)


## References

[1] N. Nonkes, S. Agaronian, E. Kanoulas, and R. Petcu, "Leveraging Graph Structures to Detect Hallucinations in Large Language Models," in Proceedings of TextGraphs-17: Graph-based Methods for Natural Language Processing, Bangkok, Thailand, Aug. 2024, pp. https://aclanthology.org/2024.textgraphs-1.7


