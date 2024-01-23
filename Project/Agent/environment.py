import torch
import numpy as np

def to_patches(x, region_size):  # x.shape = (C, H, W)
    k = x.shape[1]
    x = x.unfold(2, region_size, region_size).unfold(3, region_size, region_size)
    x = x.permute(0, 2, 3, 1, 4, 5)
    x = x.reshape(-1, k, region_size, region_size)
    return x

class Environment:
    def __init__(self, seg_model, nregions: int, image_size: int, device: str) -> None:
        self.seg_model = torch.load(seg_model, map_location=device)
        self.seg_model.eval()
        self.device = device
        self.nregions = nregions
        self.region_size = image_size // self.nregions

    def _calc_reward(self, mask: torch.Tensor) -> int:
        ''' V1: Only regions with 1s would get rewarded. No penalties for repetition.
            V2: Repeated choices penalized and regions with 1s scaled to a range of 0-200.
            V3: Regions without 1s in top 4 choices get a penalty according to their position.
            V4: V2 but the range is 0-100 and rewards are log scaled to increase importance. '''
        region_area = self.region_size**2
        ones_count = torch.sum((mask == 1).float())
        reward = (ones_count / region_area) * 200
        #reward = ((np.log(ones_count.item() + 1) / np.log(region_area + 1)) * 100)
        #reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
        return reward / self.iteration

    def reset(self, image, mask) -> None:
        region_size = image.shape[1] // self.nregions
        self.image_regions = to_patches(image[None], region_size)
        self.mask = mask.long()[None, None]
        self.region_selected = torch.zeros(self.nregions**2, dtype=bool)
        self.iteration = 0

    def possible_actions(self) -> int:
        return self.nregions**2

    def was_region_selected(self, action: int) -> bool:
        return self.region_selected[action]
                
    def step(self, action: int) -> (int, bool, int):
        assert 0 <= action < self.possible_actions()
        self.iteration += 1
        if self.was_region_selected(action):
            reward = torch.tensor(-20, dtype=torch.float32)
            return reward, False, reward  # If region selected prev, penalize agent decision by -100
        self.region_selected[action] = True
        with torch.no_grad():
            preds = self.seg_model(self.image_regions[[action]])
            preds = preds['out'][:, [1]]  # (1, 1, 256, 256)
            preds = torch.sigmoid(preds) >= 0.5
        reward = self._calc_reward(preds)
        isover = torch.all(self.region_selected)
        return reward, isover, reward

if __name__ == '__main__':  # DEBUG
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    import data
    
    assert 2048 % 4 == 0
    region_size = 2048 // 4

    transform = A.Compose([
        A.Resize(2048, 2048),
        A.Normalize(0, 1),
        ToTensorV2(),
    ])
    
    ds = getattr(data, 'BDD10K_Binary')
    ds = ds('/nas-ctm01/datasets/public', 'test', transform)
    image = ds[0]['image'].to('cuda')
    mask = ds[0]['mask'].to('cuda').long()
    print(image.shape, mask.shape)
    
    env = Environment(seg_model='../DL3RN50_BDD10KBin_4R_500E_2048.pth',
                      nregions=4,
                      image_size=2048,
                      device='cuda')
    env.reset(image, mask)
    print(env.image_regions.shape, env.mask.shape)
    #for i in range(8):
    #    import matplotlib.pyplot as plt
    #    plt.imshow(env.image_regions[i].permute(1, 2, 0))
    #    plt.savefig(f'debug-{i}.jpg')
    print('possible actions:', env.possible_actions())
    print('was region selected:', env.was_region_selected(5))
    for i in range(16):
        print('step:', i, env.step(i))