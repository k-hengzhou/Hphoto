# 人脸聚类
# 输入：人脸向量
# 输出：人脸聚类结果
# 功能：将人脸向量进行聚类，并保存到人脸聚类结果中
# 人脸聚类结果用于人脸识别，将人脸识别结果与人脸聚类结果进行匹配，找到人脸
import os
import cv2
import numpy as np
import insightface
import msgpack
from pathlib import Path
from sklearn.cluster import DBSCAN
from tqdm import tqdm
import psutil

def calculate_distance_matrix(embeddings):
    """计算特征向量之间的距离矩阵，确保所有值非负"""
    n = len(embeddings)
    distance_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            # 计算余弦相似度
            similarity = np.dot(embeddings[i], embeddings[j])
            similarity = similarity / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            # 确保相似度在[-1, 1]范围内
            similarity = np.clip(similarity, -1.0, 1.0)
            # 转换为距离：将相似度从[-1, 1]映射到[0, 1]
            distance = (1.0 - similarity) / 2.0
            # 添加一个小的epsilon避免数值误差
            distance = max(distance, 1e-10)
            distance_matrix[i, j] = distance
    
    return distance_matrix

def save_clusters(clusters, output_path):
    """保存聚类结果到文件"""
    try:
        # 将numpy数组转换为列表
        clusters_to_save = {}
        for name, paths in clusters.items():
            clusters_to_save[name] = [str(path) for path in paths]
        
        # 使用msgpack保存
        with open(output_path, 'wb') as f:
            msgpack.dump(clusters_to_save, f)
        print(f"聚类结果已保存到: {output_path}")
    except Exception as e:
        print(f"保存聚类结果失败: {str(e)}")

def load_clusters(input_path):
    """从文件加载聚类结果"""
    try:
        with open(input_path, 'rb') as f:
            clusters = msgpack.load(f)
            # 将路径字符串转换为Path对象
            return {name: [Path(p) for p in paths] for name, paths in clusters.items()}
    except Exception as e:
        print(f"加载聚类结果失败: {str(e)}")
        return {}

class FaceClusterer:
    def __init__(self, min_samples=3, eps=0.3):
        """初始化人脸聚类器
        
        Args:
            min_samples: 形成一个类别所需的最小样本数，默认3
            eps: DBSCAN的邻域大小参数，默认0.3
        """
        self.min_samples = min_samples
        self.eps = eps
        
        # 初始化人脸检测和特征提取模型
        try:
            self.face_analyzer = insightface.app.FaceAnalysis(
                providers=['CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {str(e)}")
            raise

    def extract_face_features(self, image_path):
        """从图片中提取人脸特征"""
        try:
            # 读取图片
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
            
            # BGR转RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 检测人脸
            faces = self.face_analyzer.get(img)
            if not faces:
                return None
            #去除不正常 不合理的人脸
            faces = [face for face in faces if face.det_score > 0.8]
            face = max(faces,key=lambda x:x.det_score)
            # 只返回特征向量和图片路径
            if face is not None:
                return [(face.embedding, image_path)]
            else:
                return None
            
        except Exception as e:
            return None
    def cluster_faces(self, embeddings,all_paths):
        """对特征向量进行聚类"""
        X = np.array(embeddings)
        distance = calculate_distance_matrix(X)
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric='precomputed')
        labels = clustering.fit(distance)
        labels = clustering.labels_
        result = []
        # print(labels)
        unique_labels = set(labels)
        j=0
        for label in unique_labels:
            if label == -1:
                continue
            indices = np.where(labels == label)[0]
            if len(indices) >= self.min_samples:
                result.append([all_paths[i] for i in indices])
                j+=1
        return result
    def get_clusters(self, input_dir, output_path=None):
        """获取聚类结果，可选保存到文件"""
        # 收集所有图片文件
        image_files = []
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            image_files.extend(Path(input_dir).rglob(ext))
        
        if not image_files:
            return {}
        
        # 提取所有人脸特征
        all_features = []
        all_paths = []
        
        # 移除进度条，直接遍历
        for img_path in image_files:
            while True:
                if psutil.cpu_percent(interval=1) < 50:
                    break
            features = self.extract_face_features(str(img_path))
            if features:
                for embedding, path in features:
                    all_features.append(embedding)
                    all_paths.append(path)
        
        if not all_features:
            return {}
        
        # 准备聚类数据
        X = np.array(all_features)
        
        # 使用DBSCAN进行聚类
        distance = calculate_distance_matrix(X)
        labels = self.cluster_faces(all_features)
        # 将结果组织成字典格式
        result = {}
        # 获取唯一标签
        unique_labels = set(labels)
        j=0
        for label in unique_labels:
            if label == -1:  # 跳过噪声点
                continue
                
            # 获取该类别的所有样本索引
            indices = np.where(labels == label)[0]
    
            # 只处理样本数大于3的类别
            if len(indices) >= self.min_samples:
                key = f'未知人物_{j}'
                result[key] = [all_paths[i] for i in indices]
                j+=1
        
        # 如果指定了输出路径，保存结果
        if output_path:
            save_clusters(result, output_path)
            
        return result

def main():
    # 创建聚类器实例
    clusterer = FaceClusterer(min_samples=3, eps=0.3)
    
    # 获取输入文件夹路径
    input_dir = input("请输入图片文件夹路径: ").strip()
    if not os.path.exists(input_dir):
        print("文件夹不存在!")
        return
    
    # 获取输出文件路径（可选）
    output_path = input("请输入保存结果的文件路径（直接回车跳过）: ").strip()
    output_path = output_path if output_path else None
    
    # 获取聚类结果
    clusters = clusterer.get_clusters(input_dir, output_path)
    
    if clusters:
        print("\n聚类结果:")
        for person, paths in clusters.items():
            print(f"\n{person}: {len(paths)}张照片")
            # print("图片路径:", paths)  # 如果需要查看具体路径可以取消注释
    else:
        print("未找到满足条件的人脸类别")
    
    return clusters

if __name__ == "__main__":
    main() 