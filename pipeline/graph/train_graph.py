import torch, argparse
from torcheval.metrics import MulticlassConfusionMatrix, MulticlassAccuracy, MulticlassRecall, MulticlassPrecision, BinaryRecall
from torch_geometric.utils import remove_isolated_nodes

from GAT import GAT
import utils_graph

from os.path import join as path_join
torch.set_printoptions(profile="full")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # command line args for specifying the situation
    parser.add_argument("--use-cuda", action="store_true", default=False,
                        help="Use GPU acceleration if available")
    parser.add_argument("--path", type=str, default="../data/",
                        help="Path to the data folder")
    parser.add_argument("--output_dir", type=str, default="../weights/",
                        help="Path to save model weights to")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of epochs to train the model")
    parser.add_argument("--optimizer", type=str, default="Adam",
                        choices=["SGD", "Adam"],
                        help="Which optimizer to use for training")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Learning rate for the optimizer")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="Whether to save best model weights")
    parser.add_argument("--output_txt", type=str, default="../train_graph.txt",
                        help="Path to save the output txt file")
    args = parser.parse_args()

    output_txt = args.output_txt
    with open(output_txt, "w") as f:
        f.write("")

    # for reproducibility
    utils_graph.set_seed(42)
    
    print("STARTING...  setup:")
    print(args)
    print("-" * 120)
    print("\n" * 2)
    with open(output_txt, "a") as f:
        f.write("STARTING...  setup:\n")
        f.write(str(args))
        f.write("\n")
        f.write("-" * 120)
        f.write("\n" * 2)

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

    ######## DATA LEAKAGE PREVENTION WITH SPECIFIC EDGE ATTRIBUTES ########
    train_idx = graph.train_idx # [t, ]
    val_idx = graph.val_idx # [v, ]
    edge_index = graph.edge_index.T # [N, 2]; (i, j) node pairs as rows

    # load the distances
    distances = torch.load(path_join(args.path, "distances.pt"), map_location=device)
    # get the distances corresponding to the nodes that have edges
    edge_attr = distances[edge_index[:, 0], edge_index[:, 1]] # [N, ]

    # these are all the edges between only train nodes
    train_mask = (torch.isin(edge_index[:, 0], train_idx)) & (torch.isin(edge_index[:, 1], train_idx))
    # these are all the edges only between train and/or validation
    val_mask = ((torch.isin(edge_index[:, 0], train_idx)) | (torch.isin(edge_index[:, 0], val_idx))) \
                    & ((torch.isin(edge_index[:, 1], train_idx)) | (torch.isin(edge_index[:, 1], val_idx)))

    # make all non train attributes zero
    train_edge_attr = edge_attr.detach().clone()
    train_edge_attr[~train_mask] = 0.    

    # make all non train and validation attributes zero
    val_edge_attr = edge_attr.detach().clone()
    val_edge_attr[~val_mask] = 0.
    ######## DATA LEAKAGE PREVENTION WITH SPECIFIC EDGE ATTRIBUTES ########

    # define model
    in_channels = graph.num_features
    out_channels = graph.y.shape[1] # number of columns
    hidden_channels = 32
    in_head = 2
    dropout = 0.2

    embedder_file = f"embedder_act_ReLU_opt_AdamW_lr_0.0001_bs_256_t_0.07.pt"
    embedder = torch.nn.Sequential(*[torch.nn.Linear(in_channels, in_channels), torch.nn.ReLU(), torch.nn.Linear(in_channels, 128)])
    embedder.load_state_dict(torch.load(path_join(args.output_dir, embedder_file), map_location=device)["state_dict"])
    gat = GAT(embedder, n_in=in_channels, hid=hidden_channels,
                     in_head=in_head, 
                     n_classes=out_channels, dropout=dropout)
    gat.to(device)

    # cross entropy loss -- w/ logits
    loss_func = torch.nn.BCEWithLogitsLoss()

    # evaluation metrics
    acc = MulticlassAccuracy(num_classes=4)
    conf = MulticlassConfusionMatrix(num_classes=4)
    macro_recall = MulticlassRecall(num_classes=4, average="macro")
    macro_precision = MulticlassPrecision(num_classes=4, average="macro")
    binary_recall = BinaryRecall()

    optimizer = utils_graph.get_optimizer(args.optimizer, gat, args.learning_rate)

    best_recall = 0.
    for i in range(args.epochs):

        # Train epoch and train loss; edge attributes between non train nodes are zero
        train_loss, model_output = utils_graph.train_loop(graph, gat, loss_func, optimizer, train_edge_attr)

        # Rewrite the train labels from vectors to integers
        y_pred_train, y_train = utils_graph.rewrite_labels(model_output[graph.train_idx].sigmoid().round()).long(), torch.sum(graph.y[graph.train_idx], dim=-1).long()
        
        # Validation epoch and valuation loss; edge attributes between non train/val nodes are zero
        val_loss, model_output = utils_graph.val_loop(graph, gat, loss_func, val_edge_attr)
        
        # Rewrite the labels from vectors to integers
        y_pred_val, y_val = utils_graph.rewrite_labels(model_output[graph.val_idx].sigmoid().round()).long(), torch.sum(graph.y[graph.val_idx], dim=-1).long()

        # Train and valuation confusion matrices
        train_conf = utils_graph.confusion_matrix(conf, y_pred_train, y_train)
        val_conf = utils_graph.confusion_matrix(conf, y_pred_val, y_val)

        # Train and valuation accuracy
        train_acc = utils_graph.accuracy(acc, y_pred_train, y_train)
        val_acc = utils_graph.accuracy(acc, y_pred_val, y_val)

        # Train and valuation macro recall
        train_macro_recall = utils_graph.macro_recall(macro_recall, y_pred_train, y_train)
        val_macro_recall = utils_graph.macro_recall(macro_recall, y_pred_val, y_val)

        # Train and valuation macro recall
        train_macro_precision = utils_graph.macro_recall(macro_precision, y_pred_train, y_train)
        val_macro_precision = utils_graph.macro_recall(macro_precision, y_pred_val, y_val)

        # Train and valuation binary accuracy
        binary_mask = torch.logical_or((y_train == 0), (y_train == 3))
        y_binary_train = utils_graph.rewrite_labels_binary(y_train[binary_mask])
        y_binary_pred_train = utils_graph.rewrite_labels_binary(y_pred_train[binary_mask])
        train_binary_recall = utils_graph.binary_recall(binary_recall, y_binary_pred_train, y_binary_train)
        
        binary_mask = torch.logical_or((y_val == 0), (y_val == 3))
        y_binary_val = utils_graph.rewrite_labels_binary(y_val[binary_mask])
        y_binary_pred_val = utils_graph.rewrite_labels_binary(y_pred_val[binary_mask])
        val_binary_recall = utils_graph.binary_recall(binary_recall, y_binary_pred_val, y_binary_val)

        # Print train and valuation loss
        print(f"Epoch: {i}\n\ttrain loss: {train_loss}\n\tval loss: {val_loss}")
        # Print train and valuation confusion matrices
        print(f"\ttrain confusion matrix:\n\t{train_conf.long()}\n\tval confusion matrix:\n\t{val_conf.long()}")
        # Print train and valuation accuracy
        print(f"\ttrain accuracy: {train_acc.item()}\n\tval accuracy: {val_acc.item()}")
        # Print train and valuation macro recall
        print(f"\ttrain macro recall: {train_macro_recall.item()}\n\tval macro recall: {val_macro_recall.item()}")
        # Print train and valuation macro precision
        print(f"\ttrain macro precision: {train_macro_precision.item()}\n\tval macro precision: {val_macro_precision.item()}")
        # Print train and valuation binary accuracy
        print(f"\ttrain binary recall: {train_binary_recall.item()}\n\tval binary recall: {val_binary_recall.item()}")

        with open(output_txt, "a") as f:
            f.write(f"Epoch: {i}\n\ttrain loss: {train_loss}\n\tval loss: {val_loss}\n")
            f.write(f"\ttrain confusion matrix:\n\t{train_conf.long()}\n\tval confusion matrix:\n\t{val_conf.long()}\n")
            f.write(f"\ttrain accuracy: {train_acc.item()}\n\tval accuracy: {val_acc.item()}\n")
            f.write(f"\ttrain macro recall: {train_macro_recall.item()}\n\tval macro recall: {val_macro_recall.item()}\n")
            f.write(f"\ttrain macro precision: {train_macro_precision.item()}\n\tval macro precision: {val_macro_precision.item()}\n")
            f.write(f"\ttrain binary recall: {train_binary_recall.item()}\n\tval binary recall: {val_binary_recall.item()}\n")

        if val_macro_recall.item() > best_recall:
            best_recall = val_macro_recall.item() 
            best_i = i        
            save = {
                "state_dict": gat.state_dict(),
                }
    if args.save_model:
        torch.save(save, path_join(args.output_dir, f"GAT_{best_i}.pt"))