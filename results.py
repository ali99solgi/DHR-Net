from DHR import DHR
from torch.utils.data import DataLoader
from Unet_architecture import UNet, hair_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import torch
import gc

train_images_path = "D:/DOWNLOAD/DIGITAL HAIR DATASET/train/original_images/"
train_mask_path = "D:/DOWNLOAD/DIGITAL HAIR DATASET/train/mask/"

val_images_path = "D:/DOWNLOAD/DIGITAL HAIR DATASET/val/original_images/"
val_mask_path = "D:/DOWNLOAD/DIGITAL HAIR DATASET/val/mask/"

test_images_path = "D:/DOWNLOAD/DIGITAL HAIR DATASET/test/original_images/"
test_mask_path = "D:/DOWNLOAD/DIGITAL HAIR DATASET/test/mask/"

train_dataset = hair_dataset(images_dir=train_images_path,masks_dir=train_mask_path)
val_dataset = hair_dataset(images_dir=val_images_path,masks_dir=val_mask_path)
test_dataset = hair_dataset(images_dir=test_images_path,masks_dir=test_mask_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet()
Unet_checkpoint_dir ='checkpoints/final_model.pt'
model.load_state_dict(torch.load(Unet_checkpoint_dir))
model.to(device=device)

batch_size = 1

train_data = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=False)
val_data = DataLoader(dataset=val_dataset,batch_size=batch_size,shuffle=False)
test_data = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

with torch.no_grad():
    
    train_acc = 0
    
    for batch in tqdm(train_data):

        images, true_masks = batch['images'], batch['masks']
        images = images.to(device=device)
        true_masks = true_masks.to(device=device)

        masks_pred = model(images)
        acc = cv2.absdiff(masks_pred.squeeze(1).cpu().detach().numpy(),true_masks.cpu().detach().numpy()).mean()
        train_acc += acc

        gc.collect()
        torch.cuda.empty_cache()

train_acc = 100-(train_acc*100/80)
print(train_acc)

with torch.no_grad():
    
    val_acc = 0
    
    for batch in tqdm(val_data):

        images, true_masks = batch['images'], batch['masks']
        images = images.to(device=device)
        true_masks = true_masks.to(device=device)

        masks_pred = model(images)
        acc = cv2.absdiff(masks_pred.squeeze(1).cpu().detach().numpy(),true_masks.cpu().detach().numpy()).mean()
        val_acc += acc

        gc.collect()
        torch.cuda.empty_cache()

val_acc = 100-(val_acc*100/20)
print(val_acc)

with torch.no_grad():
    
    test_acc = 0
    
    for batch in tqdm(test_data):

        images, true_masks = batch['images'], batch['masks']
        images = images.to(device=device)
        true_masks = true_masks.to(device=device)

        masks_pred = model(images)
        acc = cv2.absdiff(masks_pred.squeeze(1).cpu().detach().numpy(),true_masks.cpu().detach().numpy()).mean()
        test_acc += acc

        gc.collect()
        torch.cuda.empty_cache()

test_acc = 100-(test_acc*100/206)
print(test_acc)

image = cv2.imread("D:/DOWNLOAD/DIGITAL HAIR DATASET/test/original_images/ISIC_0011360.png")
inpainted_image = DHR(image,upper_bound=4,checkpoint='checkpoints/final_model.pt',output_path='',save=False)

plt.figure(1)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.figure(2)
plt.imshow(cv2.cvtColor(inpainted_image,cv2.COLOR_BGR2RGB))

image = cv2.imread("D:/DOWNLOAD/DIGITAL HAIR DATASET/test/original_images/ISIC_0000214.png")
inpainted_image = DHR(image,upper_bound=4,checkpoint='checkpoints/final_model.pt',output_path='',save=False)

plt.figure(3)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.figure(4)
plt.imshow(cv2.cvtColor(inpainted_image,cv2.COLOR_BGR2RGB))

image = cv2.imread("D:/DOWNLOAD/DIGITAL HAIR DATASET/test/original_images/ISIC_0014542.png")
inpainted_image = DHR(image,upper_bound=4,checkpoint='checkpoints/final_model.pt',output_path='',save=False)

plt.figure(5)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.figure(6)
plt.imshow(cv2.cvtColor(inpainted_image,cv2.COLOR_BGR2RGB))

image = cv2.imread("D:/DOWNLOAD/DIGITAL HAIR DATASET/test/original_images/ISIC_0009944.png")
inpainted_image = DHR(image,upper_bound=6,checkpoint='checkpoints/final_model.pt',output_path='',save=False)

plt.figure(7)
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.figure(8)
plt.imshow(cv2.cvtColor(inpainted_image,cv2.COLOR_BGR2RGB))
