import os
import skimage


def update_tiffs(source_path: str, target_path: str):
    sub_directories = [dir_info[0] for dir_info in os.walk(source_path)]
    images_path = []
    for sub_dir in sub_directories:
        files = os.listdir(sub_dir)
        for file in files:
            if file.endswith(".tif"):
                images_path += [os.path.join(sub_dir, file)]

    for image_path in images_path:
        target_image_path = image_path.replace(source_path, target_path)
        dir_name = os.path.dirname(target_image_path)
        if not os.path.isdir(dir_name):
            os.makedirs(dir_name)
        image = skimage.io.imread(image_path)
        skimage.io.imsave(target_image_path, image)
