import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG'
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def dataloader(filepath, log, split_file):
    left_fold = 'image_2/'
    right_fold = 'image_3/'
    disp_L = 'disp_occ_0/'
    semantic = 'semantic/'

    train = [x for x in os.listdir(os.path.join(filepath, '', left_fold)) if is_image_file(x)]
    left_train = [os.path.join(filepath, '', left_fold, img) for img in train]
    right_train = [os.path.join(filepath, '', right_fold, img) for img in train]
    left_train_disp = [os.path.join(filepath, '', disp_L, img) for img in train]
    left_train_semantic = [os.path.join(filepath, '', semantic, img) for img in train]

    val = [x for x in os.listdir(os.path.join(filepath, '', left_fold)) if is_image_file(x)]
    left_val = [os.path.join(filepath, '', left_fold, img) for img in val]
    right_val = [os.path.join(filepath, '', right_fold, img) for img in val]
    left_val_disp = [os.path.join(filepath, '', disp_L, img) for img in val]
    left_val_semantic = [os.path.join(filepath, '', semantic, img) for img in val]

    return left_train, right_train, left_train_disp, left_val, right_val, left_val_disp, train, left_train_semantic, left_val_semantic