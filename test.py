from DHR import DHR,hair_segmentation
import matplotlib.pyplot as plt
import cv2

image = cv2.imread('ISIC_0011327_original.png')
mask = cv2.imread('ISIC_0011327_mask.png')

mask_pred = hair_segmentation(image)
inpainted_image = DHR(image,output_path='ISIC_0011327_inpainted.png',save=True)

plt.figure(1) # Image
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB)) 
plt.figure(2) # Ground Truth (Hairs Mask)
plt.imshow(cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY),'gray') 
plt.figure(3) # Result of Hair Segmentation by U-net
plt.imshow(mask_pred,'gray') 
plt.figure(4) # Inpainted Image (Without Hair)
plt.imshow(cv2.cvtColor(inpainted_image,cv2.COLOR_BGR2RGB)) 
