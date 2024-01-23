import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='DL3RN50_BDD10KBin_4R_500E_2048.pth')
parser.add_argument('--dataset', default='BDD10K_Binary')
parser.add_argument('--batchsize', type=int, default=2)
parser.add_argument('--imgsize', type=int, default=2048)
parser.add_argument('--nregions', type=int, default=4)
args = parser.parse_args()

import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall
from tqdm import tqdm
import data

device = 'cuda' if torch.cuda.is_available() else 'cpu'

########################### DATA AUG ###########################

assert args.imgsize % args.nregions == 0
region_size = args.imgsize // args.nregions

transform = A.Compose([
    A.Resize(args.imgsize, args.imgsize),
    A.Normalize(0, 1),
    ToTensorV2(),
])

############################# DATA #############################

ds = getattr(data, args.dataset)
ds = ds('/nas-ctm01/datasets/public', 'test', transform)
# Image (3 color): (2, 3, 2048, 2048), Mask (Boolean): (2, 2048, 2048)
ts = torch.utils.data.DataLoader(ds, args.batchsize, num_workers=4, pin_memory=True)

############################ MODEL ############################

model = torch.load(args.model, map_location=device)

############################ LOOP ############################

metric = MulticlassJaccardIndex(ds.num_classes).to(device)
precision = BinaryPrecision().to(device)
recall = BinaryRecall().to(device)

def to_patches(x):
    # Get color channels
    k = x.shape[1]
    
    # Divide width and height into regions
    x = x.unfold(2, region_size, region_size).unfold(3, region_size, region_size)
    
    # Rearrange array to include color channels in region division
    x = x.permute(0, 2, 3, 1, 4, 5)
    
    # Get the regions of the images into a tensor
    x = x.reshape(-1, k, region_size, region_size)
    return x

for d in tqdm(ts):
    images = d['image'].to(device)  # N x C x H x W
    masks = d['mask'].to(device).long()[:, None]  # N x 1 x H x W

    images = to_patches(images)
    masks = to_patches(masks)[:, 0]

    preds = model(images)['out']
    torch.set_printoptions(precision=10, threshold=10000)
    preds = torch.softmax(preds, 1).argmax(1)
    metric.update(preds, masks)
    precision.update(preds, masks)
    recall.update(preds, masks)
    

print(f"Jaccard: {metric.compute().item()}")
print(f"Precision: {precision.compute().item()}")
print(f"Recall: {recall.compute().item()}")