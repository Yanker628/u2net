import argparse
import datetime
import os
import time

import torch
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from src import U2Net_lite_DW  # optional
from train_utils import get_params_groups, create_lr_scheduler, train_one_epoch, evaluate
from train_utils.my_dataset import CrackDataset
from train_utils.transforms import PresetTrain, PresetEval


def main(args):
    # set device && batch size
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size

    # save information with txt
    results_file = "result{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    # prepare dataset
    train_dataset = CrackDataset(args.data_path, train=True, transforms=PresetTrain([320, 320], crop_size=288))
    val_dataset = CrackDataset(args.data_path, train=False, transforms=PresetEval([320, 320]))

    train_data_size = len(train_dataset)
    eval_data_size = len(val_dataset)

    print(f"The length of training set is:{train_data_size}")
    print(f"The length of validation set is:{eval_data_size}")

    # compose data
    num_workers = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    train_data_loader = data.DataLoader(train_dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        shuffle=True,
                                        pin_memory=True,
                                        collate_fn=train_dataset.collate_fn)

    val_data_loader = data.DataLoader(val_dataset,
                                      batch_size=1,  # must be 1 when evaluate
                                      num_workers=num_workers,
                                      pin_memory=True,
                                      collate_fn=val_dataset.collate_fn)

    # set model
    model = U2Net_lite_DW()
    model.to(device)

    # set optimizer && learning rate
    params_group = get_params_groups(model, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(params_group, lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, len(train_data_loader), args.epochs,
                                       warmup=True, warmup_epochs=1)

    # use mixed precision training
    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # resume training option
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp:
            scaler.load_state_dict(checkpoint["scaler"])

    # record training parameters with tensorboard
    writer = SummaryWriter()

    # start training
    start_time = time.time()
    current_mae, current_f1 = 1.0, 0.0
    for epoch in range(args.start_epoch, args.epochs):
        mean_loss, lr = train_one_epoch(model, optimizer, train_data_loader, device, epoch,
                                        lr_scheduler=lr_scheduler, print_freq=args.print_freq, scaler=scaler)

        tags = ["training_loss", "learning_rate"]
        writer.add_scalar(tags[0], mean_loss, epoch)
        writer.add_scalar(tags[1], lr, epoch)

        save_file = {"model": model.state_dict(),
                     "optimizer": optimizer.state_dict(),
                     "lr_scheduler": lr_scheduler.state_dict(),
                     "epoch": epoch,
                     "args": args}
        if args.amp:
            save_file["scaler"] = scaler.state_dict()

        if epoch % args.eval_interval == 0 or epoch == args.epochs - 1:
            # evaluate each interval
            mae_metric, f1_metric = evaluate(model, val_data_loader, device=device)
            mae_info, f1_info = mae_metric.compute(), f1_metric.compute()
            writer.add_scalar("MAE", mae_info, epoch)
            writer.add_scalar("maxF1", f1_info, epoch)
            print(f"[epoch: {epoch}] val_MAE: {mae_info:.3f} val_maxF1: {f1_info:.3f}")
            # write into txt
            with open(results_file, "a") as f:
                # record parameters
                write_info = f"[epoch: {epoch}] train_loss: {mean_loss:.4f} lr: {lr:.6f} " \
                             f"MAE: {mae_info:.4f} maxF1: {f1_info:.4f} \n"
                f.write(write_info)

            # save_best
            if current_mae >= mae_info and current_f1 <= f1_info:
                torch.save(save_file, "save_weights/u2net_DW-200/model_best.pth")

        # only save latest 5 epoch weights
        if os.path.exists(f"save_weights/model_{epoch - 5}.pth"):
            os.remove(f"save_weights/model_{epoch - 5}.pth")

        torch.save(save_file, f"save_weights/model_{epoch}.pth")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


def parse_args():
    parser = argparse.ArgumentParser(description="pytorch u2net training")

    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument("--batch-size", default=12, type=int)
    parser.add_argument("--data-path", default="./", help="CRACK root")
    parser.add_argument("--weight-decay", default=1e-4, type=float, help="weight decay (default: 1e-4)")
    parser.add_argument("--learning-rate", default=0.001, type=float, help="initial learning rate")
    parser.add_argument("--epochs", default=1, type=int, help="number of total epochs to train")
    parser.add_argument("--start-epoch", default=0, type=int, help="start epoch")
    parser.add_argument("--amp", default="True", type=bool, help="Use torch.cuda.amp for mixed precision training")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--print-freq", default=100, type=int, help="print frequency")
    parser.add_argument("--eval-interval", default=10, type=int, help="validation interval default 1 Epochs")

    arguments = parser.parse_args()

    return arguments


if __name__ == "__main__":
    parameters = parse_args()

    if not os.path.exists("./save_weights"):
        os.mkdir("./save_weights")

    main(args=parameters)
