# 创建名人数据库
# 输入：名人图片目录
# 输出：名人数据库  
# 功能：将名人图片目录下的所有图片进行聚类，并保存到名人数据库中
# 名人数据库格式：{名人名: [人脸向量]}
# 名人名：图片所在的最后一层目录名
# 人脸向量：图片中的人脸向量
# 名人数据库用于人脸识别，将人脸识别结果与名人数据库进行匹配，找到名人

from face_organizer import FaceOrganizer
import msgpack
import numpy as np
import os
import argparse
from tqdm import tqdm
from sklearn.cluster import DBSCAN
from collections import Counter
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
# def detect_celebrity(input_path):
#     face_organizer = FaceOrganizer(model_path="../model/",update_db=False,backup_db=False)
#     faces,_ = face_organizer.detect_faces(input_path)

#     return faces
def celebrity_clustering(celebrity_db,eps=0.6,min_samples=1):
    # 将celebrity_db中key相同的提取到同一组
    new_celebrity_db = {}
    pbar = tqdm(celebrity_db.items(), desc="处理进度", unit="人")
    for name,embeddings in pbar:
        pbar.set_description(f"处理: {name}")
        # 将特征向量转换为numpy数组
        try:
            embeddings_np =np.array(embeddings)
        except:
            print(f"错误: {embeddings} 不是numpy数组")
            continue
        distance_matrix = calculate_distance_matrix(embeddings)
        # 使用DBSCAN进行聚类
        clustering = DBSCAN(
            eps=float(eps/2),
            min_samples=min_samples,
            metric='precomputed'
        ).fit(distance_matrix)
        # 获取聚类标签
        labels = clustering.labels_
        print(f"聚类标签: {labels}")
        label_counts = Counter(labels)
        print(f"标签计数: {label_counts}")
        # 对于每一类取平均值，并让最大的类的的向量位于最前面
        #找到最大类
        largest_cluster = max(label_counts, key=label_counts.get)

        print(f"最大类: {largest_cluster}")
        #计算每个类的平均向量

        for label in label_counts:
            #
            if label != -1:
                #如果最大类的人数大于3，并且该类的人数为1，则跳过该类
                if largest_cluster > 3 and label_counts[label] == 1:
                    continue
                # 取同一类的向量的均值
                if name not in new_celebrity_db:
                    new_celebrity_db[name] = []
                new_celebrity_db[name].append(np.mean(embeddings_np[labels == label], axis=0))
        # 将最大类的向量移到最前面
        new_celebrity_db[name].insert(0, new_celebrity_db[name].pop())

    return new_celebrity_db
    



def main():
    # 输入参数，包含输入 输出目录
    parser = argparse.ArgumentParser()
    parser.add_argument("--input","-i", type=str, help="输入目录",required=True)
    parser.add_argument("--output","-o", type=str, help="输出目录",required=True)
    parser.add_argument("--eps","-e", type=float, help="DBSCAN的eps参数",required=False,default=0.6)
    parser.add_argument("--min_samples","-m", type=int, help="DBSCAN的min_samples参数",required=False,default=1)
    args = parser.parse_args()
    # 遍历输入目录下和子目录下 的所有图片 并创建数据库
    celebrity_db = {}
    face_organizer = FaceOrganizer(model_path="../model/",update_db=False,backup_db=False)
    
    for root,dirs,files in os.walk(args.input):
        for file in files:
            if file.lower().endswith((".jpg",".jpeg",".png",".bmp",".webp")):
                celebrity_name = os.path.basename(os.path.dirname(file))
                file_path = os.path.join(root,file)
                file_path = os.path.join(root,file)
                celebrity_name = os.path.basename(os.path.dirname(file_path))
                print(f"处理: {celebrity_name} {file_path}")
                faces,_ = face_organizer.detect_faces(file_path)
                if faces is not None:
                    if celebrity_name not in celebrity_db:
                        celebrity_db[celebrity_name] = []
                    for face in faces:
                        celebrity_db[celebrity_name].append(face.embedding)
                    print(f"处理: {celebrity_name} 现有 {len(celebrity_db[celebrity_name])} 个人脸向量")
    new_celebrity_db = celebrity_clustering(celebrity_db,args.eps,args.min_samples)
    # 将new_celebrity_db保存到output目录下
    with open(args.output,"wb") as f:
    # 将numpy数组转换为列表以便序列化
        db_to_save = {}
        for name, embeddings in new_celebrity_db.items():
            db_to_save[name] = []
            for embedding in embeddings:
                db_to_save[name].append(embedding.tolist() if isinstance(embedding, np.ndarray) else embedding)
        msgpack.pack(db_to_save,f)
    #增加写详细的输出信息
    print("总的有多少人",len(new_celebrity_db))
    original_faces = 0
    for name,faces in celebrity_db.items():
        original_faces += len(faces)
        print(f"处理前: {name} 有 {len(faces)} 个人脸向量")
    total_faces_final = 0
    for name,faces in new_celebrity_db.items():
        total_faces_final += len(faces)
        print(f"处理完成: {name} 有 {len(faces)} 个人脸向量")
    #输出处理前后的对比
    print(f"原始人脸向量: {original_faces}")
    print(f"处理完成，总的人脸向量: {total_faces_final}")
    #输出保存文件的大小
    print(f"保存文件的大小: {os.path.getsize(args.output,)} bytes")
    



if __name__ == "__main__":
    main()