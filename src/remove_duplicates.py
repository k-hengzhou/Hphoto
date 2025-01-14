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
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import imagehash
from annoy import AnnoyIndex

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
        self.model = models.resnet18(weights=None)
        # 加载自定义权重
        if os.path.exists("model/resnet18.pth"):
            state_dict = torch.load("model/resnet18.pth")
            self.model.load_state_dict(state_dict)
        else:
        # 如果本地权重不存在，则使用预训练权重
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()
        # 移除最后的分类层
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    def extract_features(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0)
            with torch.no_grad():
                features = self.model(image)
            return features.squeeze().numpy()
        except Exception as e:
            print(f"提取特征失败: {str(e)}")
            return None
    def compare_images_deep(self, image1_path, image2_path):
        feat1 = self.extract_features(image1_path)
        feat2 = self.extract_features(image2_path)
        
        if feat1 is None or feat2 is None:
            return False
            
        # 计算余弦相似度
        similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
        return similarity >= self.similarity_threshold
    def calculate_hash(self, file_path):
        try:
            # 计算感知哈希
            hash = imagehash.average_hash(Image.open(file_path))
            return hash
        except Exception as e:
            print(f"计算哈希失败: {str(e)}")
            return None
    def build_hash_db(self,photo_db):
        self.hash_db = {}
        for item in photo_db:
            hash = self.calculate_hash(item['photo_path'])
            self.hash_db[hash] = item['photo_path']
        return photo_db
    def build_annoy_index(self, photo_db):
        # 创建 Annoy 索引
        hash_dim = 64  # 假设使用64位哈希
        self.annoy_index = AnnoyIndex(hash_dim, 'hamming')
        
        # 添加所有哈希值到索引
        for i, item in enumerate(photo_db):
            hash_val = self.calculate_hash(item['photo_path'])
            hash_binary = bin(int(str(hash_val), 16))[2:].zfill(64)  # 转换为64位二进制
            hash_array = [int(bit) for bit in hash_binary]  # 转换为整数数组
            self.annoy_index.add_item(i, hash_array)
        
        # 构建索引
        self.annoy_index.build(10)  # 10棵树
        self.path_mapping = {i: item['photo_path'] for i, item in enumerate(photo_db)}
    def find_similar_by_annoy(self, query_path, n_neighbors=10):
        # 计算查询图片的哈希
        query_hash = self.calculate_hash(query_path)
        hash_binary = bin(int(str(query_hash), 16))[2:].zfill(64)  # 转换为64位二进制
        query_array = [int(bit) for bit in hash_binary]  # 转换为整数数组
    
        # 查找最近邻
        similar_indices = self.annoy_index.get_nns_by_vector(
            query_array, 
            n_neighbors,
            search_k=-1
        )
        return [self.path_mapping[i] for i in similar_indices]
    def find_similar_by_annoy_deep(self,query_path):
        # self.build_annoy_index(photo_db)
        similar_paths = self.find_similar_by_annoy(query_path)
        finally_similar_paths = []
        for path in similar_paths:
            if self.compare_images(query_path, path):
                finally_similar_paths.append(path)
        return finally_similar_paths
    def compare_images(self, img1_path, img2_path):
        hash1 = self.calculate_hash(img1_path)
        hash2 = self.calculate_hash(img2_path)
        if hash1 is not None and hash2 is not None:
            if hash1 == hash2:
                return True
        similarity = 1-(abs(hash1-hash2)/64)
        # print(similarity)
        # similarity = self.calculate_image_similarity(img1_path, img2_path)
        if similarity >= self.similarity_threshold:
            return self.compare_images_deep(img1_path, img2_path)
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