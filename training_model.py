import os
import argparse
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.distributed as dist

import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error

from resnet_model import ResnetModel
from dataset import ChestXRayDataset
from utils import collate_fn

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

def evaluate(model, epoch, dataloader, args):
    if args.rank == 0:
        model.eval()
        pb = tqdm(dataloader, desc=f"Epoch {epoch} - Evaluating")
        predicteds = []
        gts = []
        for images, labels in pb:
            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            outputs = model(images)
            predicteds.append(outputs.cpu().numpy())
            gts.append(labels.cpu().numpy())

        predicteds = np.concatenate(predicteds, axis=0)
        gts = np.concatenate(gts, axis=0)
        score = mean_squared_error(gts, predicteds, squared=False)
        print(f"RMSE score: {score}")

        return score

    return 0

def train(model, epoch, dataloader, loss_fn, optimizer):
    model.train()
    pb = tqdm(dataloader, desc=f"Epoch {epoch} - Training")
    training_loss = []
    for images, labels in pb:
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        training_loss.append(loss.item())
        pb.set_postfix({"loss": np.array(training_loss).mean()})

def training(processor, args):
    rank = args.rank * args.processors + processor
    print('initializing...')
    dist.init_process_group(backend='gloo', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(13)

    print("Creating the model ...")
    model = ResnetModel().to(device)

    print("Defining loss and optimizer ...")
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.01)

    print("Wrapping the model ...")
    model = nn.parallel.DistributedDataParallel(model)
    
    print("Creating data loaders ...")
    trainset = ChestXRayDataset("chest_xray", "chest_xray/train.json")
    valset = ChestXRayDataset("chest_xray", "chest_xray/val.json")
    testset = ChestXRayDataset("chest_xray", "chest_xray/test.json")

    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=2,
                                               sampler=train_sampler,
                                               collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(dataset=valset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=2,
                                              collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(dataset=testset,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=2,
                                              collate_fn=collate_fn)

    epoch = 0
    best = 0
    patient = 0
    while True:
        epoch += 1
        train(model, epoch, train_loader, loss_fn, optimizer)
        if args.rank == 0:
            score = evaluate(model, epoch, val_loader, args)
            evaluate(model, epoch, test_loader, args)
            if score < best:
                patient += 1
            else:
                best = score

        if patient > 5:
            break

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--processors', default=1, type=int,
                        help='number of processors per node')
    parser.add_argument('--rank', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--root-address', type=str, required=True,
                        help='IP address of root processs')
    parser.add_argument('--root-port', type=str, required=True,
                        help='port to root process')
    parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                        help='batch size')
    args = parser.parse_args()
    args.world_size = args.processors * args.nodes
    os.environ['MASTER_ADDR'] = args.root_address
    os.environ['MASTER_PORT'] = args.root_port
    mp.spawn(training, nprocs=args.processors, args=(args,))

if __name__ == '__main__':
    main()