import torch, argparse
import torch.nn.functional as F
from torch_geometric.utils import remove_isolated_nodes
from torcheval.metrics import MulticlassAUPRC, MulticlassConfusionMatrix, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, BinaryRecall

from os.path import join as path_join
import utils_graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # command line args for specifying the situation
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--output_dir", type=str, default="../weights/",
                        help="Path to save model weights to")
    parser.add_argument("--path", type=str, default="../data/",
                        help="Path to the data folder")
    parser.add_argument("--k", type=int, default=5,
                        help="The k in kNN")
    parser.add_argument("--combined", action="store_true", default=False,
                        help="Use train and validation dataset")
    parser.add_argument("--full", action="store_true", default=False,
                        help="Use train and validation dataset and test on train")
    parser.add_argument("--output_txt", type=str, default="../kNN.txt",
                        help="Path to save the output txt file")
    args = parser.parse_args()

    # for reproducibility
    utils_graph.set_seed(42)

    output_txt = args.output_txt
    with open(output_txt, "w") as f:
        f.write("")


    print("STARTING...  setup:")
    print(args)
    print("-" * 120)
    print("\n" * 2)
    with open(output_txt, "a") as f:
        f.write("STARTING...  setup:\n")
        f.write(str(args))
        f.write("\n")
        f.write("-" * 120)
        f.write("\n\n")

    # some paramaters
    if args.use_cuda:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device("cpu")

    # load graph
    graph = torch.load(path_join(args.path, "graph.pt"), map_location=device)
    graph.to(device)

    # removing isolated nodes
    isolated = (remove_isolated_nodes(graph["edge_index"])[2] == False).sum(dim=0).item()
    print(f"Number of isolated nodes = {isolated}\n")
    with open(output_txt, "a") as f:
        f.write(f"Number of isolated nodes = {isolated}\n")

    # define model
    in_channels = graph.num_features
    out_channels = graph.y.shape[1] # number of columns

    embedder_file = f"embedder_act_ReLU_opt_AdamW_lr_0.0001_bs_256_t_0.07.pt"
    embedder = torch.nn.Sequential(*[torch.nn.Linear(in_channels, in_channels), torch.nn.ReLU(), torch.nn.Linear(in_channels, 128)])
    embedder.load_state_dict(torch.load(path_join(args.output_dir, embedder_file), map_location=device)["state_dict"])
    embedder.to(device)

    train_embeddings = embedder(graph.x[graph.train_idx])
    val_embeddings = embedder(graph.x[graph.val_idx])
    if args.combined:
        embeddings = torch.concat((train_embeddings, val_embeddings))
        test_embeddings = val_embeddings
    elif args.full:
        embeddings = torch.concat((train_embeddings, val_embeddings))
        test_embeddings = embedder(graph.x[graph.test_idx])
    else:
        embeddings = train_embeddings
        test_embeddings = val_embeddings
    
    preds = torch.zeros(len(test_embeddings), dtype=graph.y.dtype, device=graph.y.device)

    for i, emb in enumerate(test_embeddings):
        dist = F.cosine_similarity(emb, embeddings, -1)

        if args.combined:
            # do not want to be neighbours with itself, includes offset
            dist[len(train_embeddings) + i] = -1.

        # get the top k most similar embeddings
        topk = torch.topk(dist, args.k).indices
        labels = graph.y.sum(-1)[topk]

        # majority vote to get prediction
        preds[i] = torch.mode(labels, 0).values
    
    # Evaluation metrics
    acc = MulticlassAccuracy(num_classes=4)
    conf = MulticlassConfusionMatrix(num_classes=4)
    macro_recall = MulticlassRecall(num_classes=4, average="macro")
    macro_precision = MulticlassPrecision(num_classes=4, average="macro")
    binary_recall = BinaryRecall()
    macro_AUPRC = MulticlassAUPRC(num_classes=4, average="macro")

    if args.full:
        y = torch.sum(graph.y[graph.test_idx], dim=-1).long()
    else:
        y = torch.sum(graph.y[graph.val_idx], dim=-1).long()

    # Train and valuation confusion matrices
    conf_mat = utils_graph.confusion_matrix(conf, preds, y)

    # Train and valuation accuracy
    accuracy = utils_graph.accuracy(acc, preds, y)

    # Train and valuation macro recall
    m_recall = utils_graph.macro_recall(macro_recall, preds, y)

    # Train and valuation macro recall
    m_precision = utils_graph.macro_precision(macro_precision, preds, y)

    # Train and valuation binary accuracy
    binary_mask = torch.logical_or((y == 0), (y == 3))
    y_binary = utils_graph.rewrite_labels_binary(y[binary_mask])
    y_binary_pred = utils_graph.rewrite_labels_binary(preds[binary_mask])
    b_recall = utils_graph.binary_recall(binary_recall, y_binary_pred, y_binary)

    # One frame agreement
    ofa = utils_graph.k_frame_agreement(preds, y, k=1)

    # Valuation macro area under the precision-recall curve
    m_AUPRC = utils_graph.macro_AUPRC(macro_AUPRC, preds, y)

    # Print train and valuation confusion matrices
    print(f"Confusion matrix:\n\t{conf_mat.long()}")
    # Print valuation accuracy
    print(f"Accuracy: {accuracy.item()}")
    # Print valuation macro recall
    print(f"Macro recall: {m_recall.item()}")
    # Print valuation macro recall
    print(f"Macro precision: {m_precision.item()}")
    # Print valuation binary accuracy
    print(f"Binary recall: {b_recall.item()}")
    # Print valuation one frame agreement
    print(f"One frame agreement: {ofa}")
    # Print valuation macro AUPRC
    print(f"Macro AUPRC: {m_AUPRC.item()}")

    with open(output_txt, "a") as f:
        f.write(f"Confusion matrix:\n\t{conf_mat.long()}\n")
        f.write(f"Accuracy: {accuracy.item()}\n")
        f.write(f"Macro recall: {m_recall.item()}\n")
        f.write(f"Macro precision: {m_precision.item()}\n")
        f.write(f"Binary recall: {b_recall.item()}\n")
        f.write(f"One frame agreement: {ofa}\n")
        f.write(f"Macro AUPRC: {m_AUPRC.item()}\n")
        f.write("\n")
 