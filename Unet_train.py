import gc
import torch
from torch import optim , Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from Unet_architecture import UNet,hair_dataset

train_images_path = "D:/DOWNLOAD/DIGITAL HAIR DATASET/train/original_images/"
train_mask_path = "D:/DOWNLOAD/DIGITAL HAIR DATASET/train/mask/"

val_images_path = "D:/DOWNLOAD/DIGITAL HAIR DATASET/val/original_images/"
val_mask_path = "D:/DOWNLOAD/DIGITAL HAIR DATASET/val/mask/"

def dice_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-6):

    sum_dim = (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    
    return dice

def dice_loss(input: Tensor, target: Tensor):
    return 1 - dice_coeff(input, target)

def train_model(
        model,
        device,
        epochs,
        train_batch_size,
        val_batch_size,
        learning_rate,
        weight_decay):

    train_dataset = hair_dataset(images_dir=train_images_path,masks_dir=train_mask_path)
    val_dataset = hair_dataset(images_dir=val_images_path,masks_dir=val_mask_path)

    train_data = DataLoader(dataset=train_dataset,batch_size=train_batch_size,shuffle=True)
    val_data = DataLoader(dataset=val_dataset,batch_size=val_batch_size,shuffle=False)

    model.to(device=device)

    optimizer = optim.Adam(model.parameters(),lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.1, patience=2 , min_lr=1e-8)

    # Begin Training
    for epoch in range(1, epochs + 1):

        print('epoch:',epoch)
        epoch_loss = 0

        for batch in tqdm(train_data):

            images, true_masks = batch['images'], batch['masks']
            images = images.to(device=device)
            true_masks = true_masks.to(device=device)

            optimizer.zero_grad()
            masks_pred = model(images)
            loss = dice_loss(masks_pred.squeeze(1), true_masks)
            epoch_loss += loss 

            loss.backward()
            optimizer.step()
            gc.collect()
            torch.cuda.empty_cache()

        #------------------------------------------------------------------------------------------------

        val_loss = 0
        with torch.no_grad():
            
            for batch in tqdm(val_data):

                images, true_masks = batch['images'], batch['masks']
                images = images.to(device=device)
                true_masks = true_masks.to(device=device)

                masks_pred = model(images)
                loss = dice_loss(masks_pred.squeeze(1), true_masks)
                val_loss += loss

            scheduler.step(val_loss)

            gc.collect()
            torch.cuda.empty_cache()

        print('Training Loss: {:.6f} \tval Loss: {:.6f} \tlr: {:.6f}'
            .format(epoch_loss.item(),val_loss.item(),optimizer.defaults['lr']))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet()

train_model(
    model=model,
    device=device,
    epochs=50,
    train_batch_size = 10,
    val_batch_size = 10,
    learning_rate=0.1,
    weight_decay = 1e-8)

torch.save(model.state_dict(),'final_model.pt')
