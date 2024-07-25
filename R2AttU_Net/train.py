import os
import numpy as np
import torch
import torchvision
from torch import optim
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from network import R2AttU_Net

class CustomDataset(Dataset):
    def __init__(self, image_file, mask_file, transform=None):
        self.image_file = image_file
        self.mask_file = mask_file
        self.transform = transform
        
        # Open images and get frames
        self.image = Image.open(image_file).convert('RGB')
        self.mask = Image.open(mask_file).convert('L')
        
        # Ensure we are working with all frames in the TIFF
        self.image_frames = self._get_all_frames(self.image)
        self.mask_frames = self._get_all_frames(self.mask)
        
        # Make sure we have the same number of frames in both image and mask
        assert len(self.image_frames) == len(self.mask_frames), "Image and mask frames count do not match"
        
    def _get_all_frames(self, img):
        frames = []
        try:
            while True:
                frames.append(np.array(img))
                img.seek(len(frames))  # Move to next frame
        except EOFError:
            pass
        return frames

    def __len__(self):
        return len(self.image_frames)

    def __getitem__(self, idx):
        image = self.image_frames[idx]
        mask = self.mask_frames[idx]
        
        # Convert numpy arrays to tensors
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0
        
        if self.transform:
            image, mask = self.transform(image, mask)
        
        return image, mask

class Config:
    def __init__(self):
        self.img_ch = 3
        self.output_ch = 1
        self.lr = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.num_epochs = 50
        self.num_epochs_decay = 30
        self.batch_size = 1
        self.num_workers = 4
        self.log_step = 2
        self.val_step = 2
        self.model_path = './models'
        self.result_path = './result'
        self.train_path = './dataset/training.tif'
        self.train_gt_path = './dataset/training_groundtruth.tif'
        self.valid_path = './dataset/testing.tif'
        self.valid_gt_path = './dataset/testing_groundtruth.tif'
        self.model_type = 'R2AttU_Net'
        self.t = 3

def build_model(config):
    model = R2AttU_Net(img_ch=config.img_ch, output_ch=config.output_ch)
    return model

def save_results(images, masks, outputs, epoch, save_path):
    images = images.squeeze().cpu().numpy().transpose(1, 2, 0) * 255.0
    masks = masks.squeeze().cpu().numpy() * 255.0
    outputs = outputs.squeeze().cpu().numpy() * 255.0

    images = Image.fromarray(images.astype(np.uint8))
    masks = Image.fromarray(masks.astype(np.uint8))
    outputs = Image.fromarray(outputs.astype(np.uint8))

    images.save(os.path.join(save_path, f'epoch_{epoch}_image.png'))
    masks.save(os.path.join(save_path, f'epoch_{epoch}_mask.png'))
    outputs.save(os.path.join(save_path, f'epoch_{epoch}_output.png'))

def train_model(model, train_loader, valid_loader, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    criterion = torch.nn.BCELoss()

    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0

        for i, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f'Epoch [{epoch+1}/{config.num_epochs}], Loss: {epoch_loss:.4f}')

        if (epoch+1) > (config.num_epochs - config.num_epochs_decay):
            for param_group in optimizer.param_groups:
                param_group['lr'] -= (config.lr / float(config.num_epochs_decay))
            print(f'Decayed learning rate to: {optimizer.param_groups[0]["lr"]:.6f}')

        # Validation
        model.eval()
        with torch.no_grad():
            for images, masks in valid_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = torch.sigmoid(model(images))
                # Save images and predictions without displaying
                save_results(images[0], masks[0], outputs[0], epoch, config.result_path)
                
        # Clear memory
        del images, masks, outputs
        torch.cuda.empty_cache()  # If using GPU, otherwise adjust for MPS

        # Save model
        torch.save(model.state_dict(), os.path.join(config.model_path, 'latest_model.pth'))

def main():
    config = Config()

    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)

    train_dataset = CustomDataset(config.train_path, config.train_gt_path)
    valid_dataset = CustomDataset(config.valid_path, config.valid_gt_path)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

    model = build_model(config)
    train_model(model, train_loader, valid_loader, config)

if __name__ == '__main__':
    main()