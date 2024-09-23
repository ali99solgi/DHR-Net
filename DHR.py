from skimage.metrics import structural_similarity as ssim
import torch
import numpy as np
import cv2
import tensorflow as tf
import neuralgym as ng
from Unet_architecture import UNet
from inpaint_model import InpaintCAModel

def hair_segmentation(image,checkpoint='checkpoints/final_model.pt'):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet()
    Unet_checkpoint_dir = checkpoint
    model.load_state_dict(torch.load(Unet_checkpoint_dir))
    model.to(device=device)

    image = image.copy()
    dim = torch.tensor(image).dim()
    
    if dim == 3:
        image = [cv2.cvtColor(image,cv2.COLOR_BGR2RGB)]
        image = torch.tensor(np.array(image),dtype=torch.float32).permute(0,3,1,2)/255

    image = image.to(device=device)
    
    mask_pred = model(image)
    mask_pred = mask_pred.squeeze(1)

    if dim == 3:
        mask_pred = mask_pred[0]
    
    mask_pred = mask_pred.cpu().detach().numpy()
    mask_pred = mask_pred*255
    mask_pred = mask_pred > 127
    mask_pred = (mask_pred).astype(np.uint8)
    mask_pred = mask_pred*255
    
    return mask_pred

def inpaint(raw_image,mask):
    
    if len(mask.shape) == 2:

        mask = cv2.cvtColor(mask,cv2.COLOR_GRAY2RGB)

    raw_image = cv2.cvtColor(raw_image,cv2.COLOR_BGR2RGB)

    image = cv2.bitwise_or(raw_image,mask)

    checkpoint_dir = 'model_logs/release_celeba_hq_256_deepfill_v2'

    FLAGS = ng.Config('inpaint.yml')

    model = InpaintCAModel()

    assert image.shape == mask.shape

    h, w, _ = image.shape
    grid = 8
    image = image[:h//grid*grid, :w//grid*grid, :]
    mask = mask[:h//grid*grid, :w//grid*grid, :]
    print('Shape of image: {}'.format(image.shape))

    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    input_image = np.concatenate([image, mask], axis=2)

    sess_config = tf.compat.v1.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=sess_config) as sess:
        input_image = tf.constant(input_image, dtype=tf.float32)
        output = model.build_server_graph(FLAGS, input_image)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        # load pretrained model
        vars_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.train.load_variable(checkpoint_dir, from_name)
            assign_ops.append(tf.compat.v1.assign(var, var_value))
        sess.run(assign_ops)
        print('Model loaded.')
        result = sess.run(output)
        inpainted_image = result[0][:, :, ::-1]
        inpainted_image = cv2.cvtColor(inpainted_image,cv2.COLOR_RGB2BGR)
        
    return inpainted_image
    
def calculate_intra_ssim(image):

    image = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    window = []
    for i in range(0,30):
        for j in range(0,30):
            x_l = 0  + 16*i
            x_h = 48 + 16*i
            y_l = 0  + 16*j
            y_h = 48 + 16*j
            window.append(image[x_l:x_h,y_l:y_h])
       
    calc = []
    for temp in window:
        sq = []
        for i in range(0,3):
            for j in range(0,3):
                x_l = 0  + 16*i
                x_h = 16 + 16*i
                y_l = 0  + 16*j
                y_h = 16 + 16*j
                sq.append(temp[x_l:x_h,y_l:y_h])

        calc.append((1/8)*(ssim(sq[5],sq[0])+ssim(sq[5],sq[1])+ssim(sq[5],sq[2])+ssim(sq[5],sq[3])+ssim(sq[5],sq[4])+ssim(sq[5],sq[6])+ssim(sq[5],sq[7])+ssim(sq[5],sq[8])))

    intra_ssim_value = 0

    for temp in calc:
        intra_ssim_value += temp

    intra_ssim_value = intra_ssim_value/900
    
    return intra_ssim_value

def DHR(image,upper_bound=4,checkpoint='checkpoints/final_model.pt',output_path='',save=False):
    
    intra_ssim_list = []
    
    mask_pred = hair_segmentation(image,checkpoint)
    inpainted_image = inpaint(image,mask_pred)
    intra_ssim = calculate_intra_ssim(inpainted_image)
    intra_ssim_list.append(intra_ssim)

    i = 1

    while(i < upper_bound):

        mask_pred = hair_segmentation(inpainted_image)
        inpainted_image = inpaint(inpainted_image,mask_pred)
        intra_ssim = calculate_intra_ssim(inpainted_image)
        intra_ssim_list.append(intra_ssim)

        if intra_ssim_list[i] == intra_ssim_list[i-1] :
            break

        i += 1 

    if save == True:
        cv2.imwrite(output_path, inpainted_image)

    return inpainted_image
