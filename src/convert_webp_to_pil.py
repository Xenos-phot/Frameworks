from PIL import Image
import os
import parse
from glob import glob
from tqdm import tqdm

def convert_webp_to_png(webp_file_path, output_file_path):
    # Open the WebP file
    with Image.open(webp_file_path) as img:
        # Convert to RGB if necessary (Pillow handles transparency in PNGs)
        if img.mode != 'RGB':
            img = img.convert('RGBA')  # Use RGBA for images with transparency

        # Save the image as a PNG
        img.save(output_file_path, format='PNG')


for file in glob('../output/phot/**/*.*', recursive=True):
    if file.endswith('webp'):
        output_file_path = file.replace('.webp', '.png')
        convert_webp_to_png(file, output_file_path)
        os.remove(file)
        print(f'Converted {file} to {output_file_path}')
    