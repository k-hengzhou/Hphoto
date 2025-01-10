import os
import hashlib
from collections import defaultdict
import argparse
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2

class DuplicateRemover:
    """
    重复图片查找和删除工具
    输入：照片目录
    输出：重复图片列表
    功能：将照片目录下的所有图片进行相似度计算，并删除重复图片
    """
    def __init__(self, similarity_threshold=0.9):
        self.file_hashes = defaultdict(list)
        self.similar_pairs = []
        self.duplicates = []
        self.bytes_saved = 0
        self.similarity_threshold = similarity_threshold
    
    def calculate_hash(self, file_path, block_size=65536):
        hasher = hashlib.md5()
        try:
            with open(file_path, 'rb') as f:
                while True:
                    data = f.read(block_size)
                    if not data:
                        break
                    hasher.update(data)
            return hasher.hexdigest()
        except Exception:
            return None
    def compare_images(self, img1_path, img2_path):
        hash1 = self.calculate_hash(img1_path)
        hash2 = self.calculate_hash(img2_path)
        if hash1 is not None and hash2 is not None:
            if hash1 == hash2:
                return True
        similarity = self.calculate_image_similarity(img1_path, img2_path)
        if similarity >= self.similarity_threshold:
            return True
        return False
    
    def calculate_image_similarity(self, img1_path, img2_path, size=(224, 224)):
        try:
            img1 = cv2.imdecode(np.fromfile(img1_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            img2 = cv2.imdecode(np.fromfile(img2_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img1 is None or img2 is None:
                return 0.0
                
            img1 = cv2.resize(img1, size)
            img2 = cv2.resize(img2, size)
            
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            score = ssim(gray1, gray2)
            return max(0.0, min(1.0, score))
            
        except Exception:
            return 0.0
    
    def find_similar_images(self, files_to_scan):
        total = len(files_to_scan)
        
        for i in range(total):
            for j in range(i + 1, total):
                similarity = self.calculate_image_similarity(
                    files_to_scan[i],
                    files_to_scan[j]
                )
                if similarity >= self.similarity_threshold:
                    self.similar_pairs.append((
                        files_to_scan[i],
                        files_to_scan[j],
                        similarity
                    ))
                    if files_to_scan[i] not in self.duplicates:
                        self.duplicates.append(files_to_scan[i])
                        try:
                            self.bytes_saved += os.path.getsize(files_to_scan[i])
                        except:
                            pass
                    if files_to_scan[j] not in self.duplicates:
                        self.duplicates.append(files_to_scan[j])
                        try:
                            self.bytes_saved += os.path.getsize(files_to_scan[j])
                        except:
                            pass
    
    def scan_directory(self, directory, recursive=True, extensions=('.jpg', '.jpeg', '.png')):
        files_to_scan = []
        if recursive:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(extensions):
                        files_to_scan.append(os.path.join(root, file))
        else:
            for file in os.listdir(directory):
                if file.lower().endswith(extensions):
                    files_to_scan.append(os.path.join(directory, file))
        
        if not files_to_scan:
            return
        
        for file_path in files_to_scan:
            file_hash = self.calculate_hash(file_path)
            if file_hash:
                self.file_hashes[file_hash].append(file_path)
        
        for hash_value, file_paths in self.file_hashes.items():
            if len(file_paths) > 1:
                original = file_paths[0]
                duplicates = file_paths[1:]
                self.duplicates.extend(duplicates)
                try:
                    file_size = os.path.getsize(original)
                    self.bytes_saved += file_size * len(duplicates)
                except:
                    pass
                for dup in duplicates:
                    if dup in files_to_scan:
                        files_to_scan.remove(dup)
                self.duplicates.append(original)
        
        self.find_similar_images(files_to_scan)
        return self.duplicates

    def remove_duplicates(self, backup_dir=None):
        if not self.duplicates:
            return
            
        if backup_dir:
            backup_path = Path(backup_dir)
            backup_path.mkdir(parents=True, exist_ok=True)
            
        removed = 0
        failed = 0
        
        for file_path in self.duplicates:
            try:
                if backup_dir:
                    dest = backup_path / Path(file_path).name
                    counter = 1
                    while dest.exists():
                        stem = dest.stem
                        if '_' in stem:
                            base = stem.rsplit('_', 1)[0]
                        else:
                            base = stem
                        dest = backup_path / f"{base}_{counter}{dest.suffix}"
                        counter += 1
                    shutil.move(file_path, dest)
                else:
                    os.remove(file_path)
                removed += 1
            except Exception:
                failed += 1
        
        return removed, failed

def main():
    parser = argparse.ArgumentParser(description='重复和相似图片查找工具')
    parser.add_argument('--directory', '-d', type=str, required=True)
    parser.add_argument('--backup', '-b', type=str)
    parser.add_argument('--no-recursive', '-n', action='store_true')
    parser.add_argument('--dry-run', '-dr', action='store_true')
    parser.add_argument('--similarity', '-s', type=float, default=0.9)
    
    args = parser.parse_args()
    
    try:
        remover = DuplicateRemover(similarity_threshold=args.similarity)
        duplicates = remover.scan_directory(
            args.directory,
            recursive=not args.no_recursive
        )
        
        if not args.dry_run and duplicates:
            remover.remove_duplicates(args.backup)
            
    except Exception as e:
        return 1
    return 0

if __name__ == "__main__":
    exit(main()) 