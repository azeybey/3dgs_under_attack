import os
from PIL import Image
import argparse

def crop_to_square(image):
    """Crop an image to a square at the center."""
    width, height = image.size
    min_dim = min(width, height)
    
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = (width + min_dim) // 2
    bottom = (height + min_dim) // 2
    
    image = image.crop((left, top, right, bottom))
    return image

def is_black_image(image):
    """Check if the image is completely black."""
    # Convert image to grayscale for simplicity in checking
    grayscale = image.convert('L')
    # Check if all pixels are black (i.e., value of 0)
    if grayscale.getextrema() == (0, 0):
        return True
    return False

def process_images(directory):
    """Process images in a given directory."""
    # List all files in the directory
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(directory, filename)
            with Image.open(file_path) as img:
                print(f'Processing {filename}...')
                
                # Check if the image is completely black
                if is_black_image(img):
                    print(f'Image {filename} is completely black. Skipping...')
                    continue

                # Check if the image is square
                if img.width != img.height:
                    print(f'Image {filename} is not square. Cropping...')
                    img = crop_to_square(img)
                
                # Save the cropped image, or you could overwrite the original
                # by using the same file_path
                save_path = os.path.join(directory, filename)
                img.save(save_path)
                print(f'Saved cropped image as {save_path}')

def crop_image(input_path, crop_margin):

    for filename in os.listdir(input_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_path, filename)
            with Image.open(file_path) as img:
                # Get the original dimensions of the image
                width, height = img.size
                
                # Define the left, upper, right, and lower pixel coordinates to crop out 5 pixels from each side
                left = crop_margin
                upper = crop_margin
                right = width - crop_margin
                lower = height - crop_margin
                
                # Crop the image
                cropped_img = img.crop((left, upper, right, lower))
                
                save_path = os.path.join(input_path, f"_{filename}")
                cropped_img.save(save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory_path', type=str, default="hydrant")
    parser.add_argument('--true_label_name', type=str, default="hydrant")
    args = parser.parse_args()

    process_images(args.directory_path)



# Example usage
#directory_path = 'data/apple/images'

#crop_margin = 5  
#crop_image(directory_path, 1)

#process_images(directory_path)
