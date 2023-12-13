import os

import torch
from torch.utils import data

from src import U2Net
from train_utils import evaluate
from train_utils.my_dataset import CrackDataset
from train_utils.transforms import PresetEval


def main(args):
    # set device && check pth
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    assert os.path.exists(args.weights), f"weights {args.weights} not found."

    # prepare dataset
    val_dataset = CrackDataset(args.data_path, train=False, transforms=PresetEval([320, 320]))

    # compose data
    num_workers = 8
    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,  # must be 1
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      shuffle=False,
                                      collate_fn=val_dataset.collate_fn)

    # set model
    model = U2Net()
    pretrain_weights = torch.load(args.weights, map_location='cpu')
    if "model" in pretrain_weights:
        model.load_state_dict(pretrain_weights["model"])
    else:
        model.load_state_dict(pretrain_weights)
    model.to(device)

    mae_metric, f1_metric = evaluate(model, val_data_loader, device=device)
    print(mae_metric, f1_metric)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch u2net validation")

    parser.add_argument("--data-path", default="./", help="Crack root")
    parser.add_argument("--weights", default="./save_weights/model_best.pth")
    parser.add_argument("--device", default="cuda:0", help="training device")
    parser.add_argument("--print-freq", default=10, type=int, help='print frequency')

    arguments = parser.parse_args()

    return arguments


if __name__ == '__main__':
    parameters = parse_args()
    main(parameters)
