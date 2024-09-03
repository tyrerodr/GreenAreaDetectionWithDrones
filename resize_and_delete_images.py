import os
from PIL import Image

def resize_images(input_dir, output_dir, size=(672, 672)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        if os.path.isfile(input_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
            try:
                with Image.open(input_path) as img:
                    img = img.resize(size, Image.LANCZOS)
                    img.save(output_path) 
                    print(f"Save: {output_path}")
            except Exception as e:
                print(f"Error {input_path}: {e}")

def delete_non_matching_images(dir1, dir2):
    images_dir1 = set(os.listdir(dir1))
    images_dir2 = set(os.listdir(dir2))
    
    non_matching_images = images_dir2 - images_dir1
    
    for image in non_matching_images:
        image_path = os.path.join(dir2, image)
        try:
            os.remove(image_path)
            print(f"Deleted: {image_path}")
        except Exception as e:
            print(f"Error {image_path}: {e}")


input_dir1 = "C:\\Utils\\GreenAreaDetectionWithDrones\\masks\\original"
input_dir2 = "C:\\Utils\\GreenAreaDetectionWithDrones\\images\\original"
output_dir1 = "C:\\Utils\\GreenAreaDetectionWithDrones\\masks\\redim"
output_dir2 = "C:\\Utils\\GreenAreaDetectionWithDrones\\images\\redim"


resize_images(input_dir1, output_dir1)
resize_images(input_dir2, output_dir2)

# delete_non_matching_images(dir1, dir2)
