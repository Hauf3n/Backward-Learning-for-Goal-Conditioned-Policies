import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def eval_recon_imgs(dataset, rm, args, num_imgs=16):
    
    dataloader = DataLoader(dataset, num_workers=args.workers, batch_size=num_imgs, shuffle=True, drop_last=True)
    recon_comparison = None
    for i, (s,_,_) in enumerate(dataloader):
        state = s.clone().detach().to(args.device).to(args.dtype)
        reconstruction = rm(state)[0].detach().cpu().numpy()
        recon_comparison = draw_reconstruction(s.numpy().transpose(0,2,3,1), reconstruction.transpose(0,2,3,1), num_imgs)
        break
    return recon_comparison

def draw_reconstruction(original_images, reconstructed_images, 
                         num_images=16, num_columns=4, offset_rows=5, offset_columns=10):

    # Ensure that we have enough images to plot
    num_images = min(num_images, len(original_images), len(reconstructed_images))

    # Calculate the number of rows needed based on the number of columns
    num_rows = (num_images + num_columns - 1) // num_columns

    # Get the maximum image height among original and reconstructed images
    max_image_height = max(img.shape[0] for img in original_images + reconstructed_images)

    # Calculate the total offset height for each comparison
    total_offset_height = offset_rows * (num_rows - 1)

    # Calculate the canvas size
    canvas_width = 0  # Initialize canvas width to zero

    for i in range(num_images):
        pair_width = original_images[i].shape[1] + reconstructed_images[i].shape[1] + offset_columns
        canvas_width = max(canvas_width, pair_width)

    canvas_height = (max_image_height + total_offset_height) * num_rows

    # Create a canvas to draw the images
    canvas = np.ones((canvas_height, canvas_width * num_columns, 3)) * 255

    for i in range(num_images):
        row = i // num_columns
        col = i % num_columns

        # Calculate the starting and ending indices for each image in the canvas
        start_row = (max_image_height + offset_rows) * row
        end_row = start_row + original_images[i].shape[0]

        # Calculate the starting and ending indices for the original image in the canvas
        if col == 0:
            start_col_orig = canvas_width * col
        else:
            start_col_orig = canvas_width * col + offset_columns
        end_col_orig = start_col_orig + original_images[i].shape[1]

        # Calculate the starting and ending indices for the reconstructed image in the canvas
        start_col_recon = start_col_orig + original_images[i].shape[1] 
        end_col_recon = start_col_recon + reconstructed_images[i].shape[1]

        # Place the original image on the canvas
        canvas[start_row:end_row, start_col_orig:end_col_orig] = original_images[i]

        # Place the reconstructed image on the canvas
        canvas[start_row:end_row, start_col_recon:end_col_recon] = reconstructed_images[i]
        
    return canvas