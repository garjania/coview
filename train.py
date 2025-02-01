import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import datasets
from models import FrameEncoder

def extend_dim_if_needed(x):
    if x.ndim == 4:
        x = x.unsqueeze(0)
    return x

def get_dataloaders(args):
    
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.dataset == "simple":
        dataset = datasets.SimpleVideoDataset(
            root=args.train_dir,
            index_slice=args.slice,
            transform=transform,
        )
        dataloader = DataLoader(dataset=dataset, batch_size=args.frame_len, shuffle=True)
    elif args.dataset == "sequential":
        dataset = datasets.SequentialVideoDataset(
            root=args.train_dir,
            frame_len=args.frame_len,
            index_slice=args.slice,
            transform=transform,
        )
        dataloader = DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=True)
    return dataloader

def train_one_epoch(args, encoder, dataloader, optimizer, epoch):
    encoder.train()
    total_loss = []
    for step, (images, frame_ids) in enumerate(dataloader):
        
        images = extend_dim_if_needed(images)
        frame_ids = extend_dim_if_needed(frame_ids)
        images = images.to(args.device)
        frame_ids = frame_ids.to(args.device)
        
        _, loss = encoder(images, frame_ids, return_loss=True)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if args.print_every != -1 and step % args.print_every == 0:
            print(f"Loss at step {step} of epoch {epoch}: {loss.item()}")
        total_loss.append(loss.item())
        
    return total_loss

def main(args):
    dataloader = get_dataloaders(args)
    encoder = FrameEncoder()
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        loss = train_one_epoch(args, encoder, dataloader, optimizer, epoch)
        print(f"Average loss at epoch {epoch}: {sum(loss) / len(loss)}")
        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            torch.save(encoder.state_dict(), output_dir / f"encoder_{epoch}.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Dataset arguments
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="sequential")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--slice", type=tuple, default=None)
    parser.add_argument("--output_dir", type=str, default="output")
    # Training arguments
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--frame_len", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    # Logging arguments
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--print_every", type=int, default=-1)
    
    args = parser.parse_args()
    
    main(args)
