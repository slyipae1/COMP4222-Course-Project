import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from torch_geometric.data import Data

import argparse
import numpy as np
from dataloader import SentenceLabelDataset
from utils_graph import *
from os.path import join as path_join

from torch_geometric.utils import to_undirected
import os



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--path", type=str, default="../data/",
                        help="Path to data")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                        help="Name of model used to embed sentences")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Similarity threshold to form an edge")
    parser.add_argument("--distances", action="store_true", default=False,
                        help="Use pre-calculated distances matrix")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output_txt", type=str, default="../make_graph.txt",
                        help="Path to save the output txt file")
    args = parser.parse_args()

    output_txt = args.output_txt
    with open(output_txt, "w") as f:
        f.write("")

    # for reproducibility
    set_seed(42)

    # use cuda if cuda is available
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if args.use_cuda:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")
    print("Device:", device)    

    # load the data from root folder
    full_dataset = SentenceLabelDataset(args.path)
    dataloader = DataLoader(full_dataset, batch_size=1,
                            shuffle=True, num_workers=args.num_workers)


    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)
    model = BertModel.from_pretrained(args.model_name)
    model.to(device)

    print("Calculating node features...")
    node_features = get_embeddings(model, tokenizer, dataloader, device)
    print("Node features shape:", node_features.shape, "\n")
    with open(output_txt, "a") as f:
        f.write("Node features shape:")
        f.write(str(node_features.shape))
        f.write("\n")

    print("Getting labels...")
    labels = get_labels(dataloader).to(device)
    print("Labels shape:", labels.shape, "\n")
    with open(output_txt, "a") as f:
        f.write("Labels shape:")
        f.write(str(labels.shape))
        f.write("\n")

    # if pre-calculated distances; use those
    if args.distances:
        distances = torch.load(path_join(args.path, "distances.pt"), map_location=device)
    else:
        distances = None

    print("Calculating edges...")
    distances, edge_index = get_edge_index(node_features, distances=distances, threshold=args.threshold)
    edge_index = edge_index.t().contiguous().to(device)
    print("Edge index shape:", edge_index.shape, "\n")
    with open(output_txt, "a") as f:
        f.write("Edge index shape:")
        f.write(str(edge_index.shape))
        f.write("\n")

    # all possible indices
    idx = np.arange(len(node_features))
    num_indices = len(idx)
    np.random.shuffle(idx)

    # get train/val/test sizes -> 70 15 15
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15

    # Calculate the number of indices for each set
    num_train = int(train_ratio * num_indices)
    num_val = int(val_ratio * num_indices)
    num_test = num_indices - num_train - num_val

    # Split the indices into train, validation, and test sets
    train_indices = idx[:num_train]
    val_indices = idx[num_train:num_train + num_val]
    test_indices = idx[num_train + num_val:]

    data = Data(x=node_features, y=labels, edge_index=edge_index)
    data.edge_index = to_undirected(data.edge_index)

    data.train_idx = torch.tensor(train_indices, dtype=torch.long)
    data.val_idx = torch.tensor(val_indices, dtype=torch.long)
    data.test_idx = torch.tensor(test_indices, dtype=torch.long)

    print("Data is directed:", data.is_directed())

    print("Final dataloader:", data)
    print("Saved graph and distances in ", args.path)

    with open(output_txt, "a") as f:
        f.write("Data is directed:")
        f.write(str(data.is_directed()))
        f.write("\n")
        f.write("Final dataloader:")
        f.write(str(data))
        f.write("\n")
        f.write("Saved graph and distances in ")
        f.write(str(args.path))
        f.write("\n")

    torch.save(data, path_join(args.path, "graph.pt"))
    torch.save(distances, path_join(args.path, "distances.pt"))