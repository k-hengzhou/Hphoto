import os
import numpy as np
import pickle
from sklearn.cluster import DBSCAN
from collections import Counter
from tqdm import tqdm
from datetime import datetime

def load_face_db(db_path):
    """加载人脸数据库"""
    if os.path.exists(db_path):
        try:
            with open(db_path, 'rb') as f:
                db = pickle.load(f)
            print(f"已加载人脸数据库，包含 {len(db)} 个人脸")
            return db
        except Exception as e:
            print(f"加载人脸数据库失败: {str(e)}")
            return {}
    return {}

def save_face_db(db_path, face_db, backup=True):
    """保存人脸数据库，可选是否备份原文件"""
    try:
        # if backup and os.path.exists(db_path):
        #     # 创建备份文件
        #     backup_path = f"{db_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
        #     os.rename(db_path, backup_path)
        #     print(f"已创建备份文件: {backup_path}")
        db_to_save = {}
        for name,faces in face_db.items():
            db_to_save[name] = []
            for face in faces:
                embedding=face
                db_to_save[name].append(embedding.astype(np.float32) if not isinstance(embedding, np.ndarray) else embedding)
        with open(db_path, 'wb') as f:
            pickle.dump(db_to_save, f)
        print(f"人脸数据库已保存: {db_path}")
    except Exception as e:
        print(f"保存人脸数据库失败: {str(e)}")

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

def process_vectors(embeddings_array, method='dbscan', eps=0.3, min_samples=2):
    """处理特征向量组，支持多种处理方法
    
    Args:
        embeddings_array: 特征向量数组
        method: 处理方法，可选：
            - 'dbscan': DBSCAN聚类后取最大类平均
            - 'kmeans': K-means聚类后取最大类平均
            - 'mean': 直接取平均值
            - 'median': 取中位数向量（与平均向量最接近的向量）
            - 'max_sim': 取与其他向量相似度之和最大的向量
    """

    if len(embeddings_array) == 1:    
        result_vector = []
        result_vector.append(embeddings_array[0])
        return result_vector
    
    if method == 'mean':
        # 直接取平均值
        mean_vector = np.mean(embeddings_array, axis=0)
        return mean_vector / np.linalg.norm(mean_vector)
    
    elif method == 'median':
        # 计算平均向量
        mean_vector = np.mean(embeddings_array, axis=0)
        mean_vector = mean_vector / np.linalg.norm(mean_vector)
        # 找出与平均向量最接近的向量
        similarities = np.array([
            np.dot(vec, mean_vector) for vec in embeddings_array
        ])
        median_idx = np.argmax(similarities)
        return embeddings_array[median_idx]
    
    elif method == 'max_sim':
        # 计算每个向量与其他向量的相似度之和
        total_similarities = np.zeros(len(embeddings_array))
        for i, vec1 in enumerate(embeddings_array):
            for j, vec2 in enumerate(embeddings_array):
                if i != j:
                    total_similarities[i] += np.dot(vec1, vec2)
        # 返回相似度之和最大的向量
        best_idx = np.argmax(total_similarities)
        return embeddings_array[best_idx]
    
    elif method == 'kmeans':
        from sklearn.cluster import KMeans
        # 估计聚类数（最多为样本数的一半）
        n_clusters = min(len(embeddings_array) // 2, 1)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(embeddings_array)
        # 找出最大的类
        labels = kmeans.labels_
        label_counts = Counter(labels)
        largest_cluster = max(label_counts, key=lambda x: label_counts[x])
        # 取最大类的平均值
        largest_cluster_vectors = embeddings_array[labels == largest_cluster]
        mean_vector = np.mean(largest_cluster_vectors, axis=0)
        return mean_vector / np.linalg.norm(mean_vector)
    
    else:  # method == 'dbscan'
        # 计算距离矩阵
        distance_matrix = calculate_distance_matrix(embeddings_array)
        # 使用DBSCAN进行聚类
        clustering = DBSCAN(
            eps=eps/2,
            min_samples=min_samples,
            metric='precomputed'
        ).fit(distance_matrix)
        # 获取聚类标签
        labels = clustering.labels_
        label_counts = Counter(labels)
        print(f"聚类标签: {labels}")
        print(f"标签计数: {label_counts}")
        # 对于每一类取平均值，并让最大的类的的向量位于最前面
        #找到最大类
        largest_cluster = max(label_counts, key=label_counts.get)
        print(f"最大类: {largest_cluster}")
        #计算每个类的平均向量
        mean_vectors = []
        for label in label_counts:
            #
            if label != -1:
                mean_vectors.append(np.mean(embeddings_array[labels == label], axis=0))
                if label == largest_cluster:
                    # 将最大类的向量移到最前面
                    mean_vectors.insert(0, mean_vectors.pop())
        return mean_vectors
def clean_face_database(db_path, output_path=None, method='dbscan', eps=0.3, min_samples=2, backup=True):
    """使用聚类方法清理人脸数据库"""
    print(f"\n开始清理人脸数据库...")
    print(f"输入文件: {db_path}")
    print(f"输出文件: {output_path or db_path}")
    print(f"参数设置: eps={eps}, min_samples={min_samples}")
    start_time = datetime.now()
    
    # 加载数据库
    face_db = load_face_db(db_path)
    if not face_db:
        print("数据库为空或加载失败")
        return
    
    # 按基础名字（不含数字）分组
    name_groups = {}
    for name, embeddings in face_db.items():
        # 获取基础名字（去掉_后的数字）
        base_name = name.split('_')[0]
        if base_name not in name_groups:
            name_groups[base_name] = []
        
        # 将向量添加到对应的组
        if not isinstance(embeddings, list):
            embeddings = [embeddings]
        name_groups[base_name].extend(embeddings)
    
    cleaned_db = {}
    total_vectors = 0
    cleaned_vectors = 0
    skipped_faces = []
    merged_faces = []
    
    # 创建进度条
    pbar = tqdm(name_groups.items(), desc="处理进度", unit="人")
    
    # 处理每个人的特征向量组
    for base_name, embeddings in pbar:
        total_vectors += len(embeddings)
        pbar.set_description(f"处理: {base_name}")
        
        # 将特征向量转换为numpy数组
        print(f"处理数量: {len(embeddings)}")
        try:
            embeddings_array = np.array(embeddings)
        except Exception as e:
            print(f"转换失败: {str(e)}")
            continue

        try:
            # 使用选择的方法处理向量
            print(f"处理数量: {len(embeddings_array)}")    
            result_vector = process_vectors(
                embeddings_array,
                method=method,
                eps=eps,
                min_samples=min_samples
            )
            print(f"处理结果: {len(result_vector)}")
            if len(result_vector) == 0:
                continue
            cleaned_db[base_name] = result_vector
            print("base_name:",base_name,"len(result_vector):",len(result_vector))
            cleaned_vectors += len(result_vector)
            
        except Exception as e:
            print(f"处理 {base_name} 时出错: {str(e)}")
            # 如果处理失败，使用平均值
            mean_vector = np.mean(embeddings_array, axis=0)
            mean_vector = mean_vector / np.linalg.norm(mean_vector)
            print("base_name:",base_name,"len(mean_vector):",len(mean_vector))
            cleaned_db[base_name] = mean_vector
            cleaned_vectors += 1
            continue
    
    # 保存清理后的数据库
    save_path = output_path or db_path
    if output_path:
        save_face_db(save_path, cleaned_db, backup=False)
    else:
        save_face_db(save_path, cleaned_db, backup=backup)
    
    # 计算处理时间
    elapsed_time = datetime.now() - start_time
    
    # 统计聚类结果
    original_names = list(face_db.keys())
    cleaned_names = list(cleaned_db.keys())
    
    # 计算大小变化
    original_size = os.path.getsize(db_path) / 1024  # KB
    new_size = os.path.getsize(save_path) / 1024  # KB
    
    # 打印简洁的统计信息
    str  = f"聚类结果统计：\n"
    str += f"人数变化: {len(original_names)} -> {len(cleaned_names)}\n"
    str += f"向量总数: {total_vectors} -> {cleaned_vectors}\n"
    str += f"文件大小: {original_size:.1f}KB -> {new_size:.1f}KB\n"
    str += f"处理时间: {elapsed_time}\n"
    print("\n" + "="*50)
    print(str)
    print("="*50)
    return str

def main():
    import argparse
    parser = argparse.ArgumentParser(description='人脸数据库清理工具')
    
    parser.add_argument('--db-path', '-d',
                       type=str,
                       default='known_faces.pkl',
                       help='输入数据库文件路径')
    
    parser.add_argument('--output', '-o',
                       type=str,
                       help='输出数据库文件路径，默认覆盖输入文件')
    
    parser.add_argument('--eps', '-e',
                       type=float,
                       default=0.3,
                       help='DBSCAN聚类的邻域半径')
    
    parser.add_argument('--min-samples', '-m',
                       type=int,
                       default=2,
                       help='DBSCAN聚类的最小样本数')
    
    parser.add_argument('--no-backup', '-n',
                       action='store_true',
                       help='不创建数据库备份文件（仅在未指定输出路径时有效）')
    
    parser.add_argument('--method', '-me',
                       type=str,
                       default='dbscan',
                       choices=['dbscan', 'kmeans', 'mean', 'median', 'max_sim'],
                       help='向量处理方法')
    
    args = parser.parse_args()
    
    try:
        clean_face_database(
            db_path=args.db_path,
            output_path=args.output,
            method=args.method,
            eps=args.eps,
            min_samples=args.min_samples,
            backup=not args.no_backup
        )
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main()) 