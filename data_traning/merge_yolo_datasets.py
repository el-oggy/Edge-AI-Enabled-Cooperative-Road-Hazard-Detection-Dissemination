import os
import shutil
import glob
import hashlib

INPUT_DIR = r'c:\Users\adars\OneDrive\Desktop\PFPD-ll\data_traning\raw_datasets'
OUTPUT_DIR = r'c:\Users\adars\OneDrive\Desktop\PFPD-ll\data_traning\unified_yolov8_dataset'

IMAGE_OUT_TRAIN = os.path.join(OUTPUT_DIR, 'images', 'train')
IMAGE_OUT_VAL = os.path.join(OUTPUT_DIR, 'images', 'val')
LABEL_OUT_TRAIN = os.path.join(OUTPUT_DIR, 'labels', 'train')
LABEL_OUT_VAL = os.path.join(OUTPUT_DIR, 'labels', 'val')

UNIFIED_CLASSES = {
    0: 'pothole',
    1: 'crack_longitudinal',
    2: 'crack_transverse',
    3: 'crack_alligator',
    4: 'construction_zone',
    5: 'accident_vehicle',
    6: 'road_obstruction',
    7: 'traffic_cone',
    8: 'barricade',
    9: 'debris'
}

SEEN_HASHES = set()

def setup_directories():
    for d in [IMAGE_OUT_TRAIN, IMAGE_OUT_VAL, LABEL_OUT_TRAIN, LABEL_OUT_VAL]:
        os.makedirs(d, exist_ok=True)

def get_image_hash(image_path):
    with open(image_path, "rb") as f:
        file_hash = hashlib.md5()
        chunk = f.read(8192)
        while chunk:
            file_hash.update(chunk)
            chunk = f.read(8192)
    return file_hash.hexdigest()

def create_data_yaml():
    yaml_path = os.path.join(OUTPUT_DIR, 'data.yaml')
    with open(yaml_path, 'w') as f:
        # For YOLO, it's safer to use relative paths if running from the output dir,
        # or absolute paths.
        abs_path = os.path.abspath(OUTPUT_DIR).replace('\\', '/')
        f.write(f"path: {abs_path}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n\n")
        f.write("names:\n")
        for k, v in sorted(UNIFIED_CLASSES.items()):
            f.write(f"  {k}: {v}\n")
    print(f"Created {yaml_path}")

def process_yolo_dataset(dataset_path, split_mapping={'train': 'train', 'valid': 'val', 'test': 'val'}):
    if not os.path.exists(dataset_path):
        return
        
    for in_split, out_split in split_mapping.items():
        img_dir = os.path.join(dataset_path, in_split, 'images')
        lbl_dir = os.path.join(dataset_path, in_split, 'labels')
        
        if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
            continue
            
        print(f"Processing {dataset_path} split: {in_split} -> {out_split}")
        
        for img_path in glob.glob(os.path.join(img_dir, '*.jpg')):
            filename = os.path.basename(img_path)
            lbl_path = os.path.join(lbl_dir, filename.replace('.jpg', '.txt'))
            
            if not os.path.exists(lbl_path):
                continue
                
            img_hash = get_image_hash(img_path)
            if img_hash in SEEN_HASHES:
                continue
            SEEN_HASHES.add(img_hash)
            
            # Since BharatPotHole uses '0' for pothole, and our unified taxonomy uses '0' for pothole, 
            # we don't need to remap classes for this specific dataset.
            # We just copy the files over.
            img_dest = os.path.join(IMAGE_OUT_TRAIN if out_split == 'train' else IMAGE_OUT_VAL, filename)
            lbl_dest = os.path.join(LABEL_OUT_TRAIN if out_split == 'train' else LABEL_OUT_VAL, filename.replace('.jpg', '.txt'))
            
            shutil.copy(img_path, img_dest)
            shutil.copy(lbl_path, lbl_dest)

if __name__ == "__main__":
    setup_directories()
    
    # Process the BharatPotHole dataset
    # Depending on how the zip was created, it might be nested
    possible_paths = [
        os.path.join(INPUT_DIR, 'BharatPotHole', 'BharatPotHole'),
        os.path.join(INPUT_DIR, 'BharatPotHole'),
        INPUT_DIR
    ]
    
    processed = False
    for p in possible_paths:
        if os.path.exists(os.path.join(p, 'train', 'images')):
            process_yolo_dataset(p)
            processed = True
            break
            
    if not processed:
        print("Could not find a valid YOLO train/images folder in", INPUT_DIR)
            
    create_data_yaml()
    print("Done merging datasets.")
