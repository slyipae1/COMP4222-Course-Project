import torch, argparse
from torcheval.metrics import MulticlassF1Score, MulticlassConfusionMatrix
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel

import utils
from baselines import MisinformationMLP, MisinformationCrossEncoder
from dataloader import SentenceLabelDataset

from os.path import join as path_join


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # command line args for specifying the situation
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use GPU acceleration if available')
    parser.add_argument('--model', type=str, default='mlp',
                        choices=["mlp", "cross_encoder"],
                        help='Model type to train and evaluate.')
    parser.add_argument('--path', type=str, default='data/full_data.csv',
                        help="Path to the data folder")
    parser.add_argument('--output_dir', type=str, default='weights/',
                        help="Path to save model weights to")
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Number of sentences to use in a batch')
    parser.add_argument('--num-workers', type=int, default=4,
                        help="Number of cores to use when loading the data")
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs to train the model')
    parser.add_argument('--no-progress-bar', action='store_true',
                        help='Hide the progress bar during training loop')
    parser.add_argument('--optimizer', type=str, default="Adam",
                        choices=["SGD", "Adam"],
                        help='Which optimizer to use for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument("--model_embed", type=str, default="bert-base-uncased",
                        help="Name of model used to embed sentences")
    args = parser.parse_args()

    # for reproducibility
    torch.manual_seed(42)
    
    print("STARTING...  setup:")
    print(args)
    print("-" * 120)
    print("\n" * 2)

    # some paramaters
    if args.use_cuda:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    else:
        device = torch.device('cpu')

    use_tqdm = not args.no_progress_bar

    # instantiate model
    if args.model == "mlp":
        s_in, s_out = 1536, 3
        model = MisinformationMLP(s_in, s_out)
    else:
        model = MisinformationCrossEncoder()
    model.to(device)

    # Load pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_embed)
    model_embed = BertModel.from_pretrained(args.model_embed)
    model_embed.to(device)

    # load the data from root folder
    # VERY HACKY WAY OF DOING THIS, TRY TO IMPROVE!
    train_dataset = SentenceLabelDataset(args.path, limit=1000)
    val_dataset = SentenceLabelDataset(args.path, limit=200)
    val_dataset.data = train_dataset.data.iloc[800:]
    train_dataset.data = train_dataset.data.head(800)

    # TODO get the queries also...
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=args.num_workers)

    # cross entropy loss -- w/ logits
    loss_func = torch.nn.BCELoss()

    # we needed to use this metric, probably only in validation
    metric = MulticlassF1Score()
    confusion = MulticlassConfusionMatrix()

    # optimizer; change
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    else:
        # uhh, that should be impossible
        ...

    best_val = 0.

    for i in range(args.epochs):
        train_loss = utils.train_loop(train_dataloader, model, model_embed, tokenizer, loss_func, optimizer, device, use_tqdm=use_tqdm)
        val_loss, val_metric, confmat = utils.val_loop(val_dataloader, model, model_embed, tokenizer, loss_func, device, metric, confusion, use_tqdm=use_tqdm)
        print(f'Epoch: {i}\n\ttrain: {train_loss}\n\tval: {val_loss}\n\n')
        print('F1 score: ', val_metric.item(), '\Confusion matrix: ', confmat.item(), '\n\n')
        if best_val < val_metric.item():
            best_val = val_metric.item()
            # Save the model - with your model name!
            save = {
            'state_dict': model.state_dict(),
            }
            torch.save(save, path_join(args.output_dir, f"{args.model}_{str(i)}.pt"))