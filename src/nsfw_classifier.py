import os
from pathlib import Path
from nudenet import NudeDetector
import cv2
import numpy as np
default_nsfw_class = ["FEMALE_GENITALIA_COVERED",
                        "BUTTOCKS_EXPOSED",
                        "FEMALE_BREAST_EXPOSED",
                        "FEMALE_GENITALIA_EXPOSED",
                        "ANUS_EXPOSED",
                        "FEET_EXPOSED",                        
                        "ANUS_COVERED",
                        "FEMALE_BREAST_COVERED",
                        "BUTTOCKS_COVERED",
                        ]

class NSFWClassifier:
    def __init__(self,unsafe_threshold=0.6,
                 model_path="D:/project/小程序/model/640m.onnx",
                 providers=['CUDAExecutionProvider'],
                 nsfw_class=default_nsfw_class):
        """初始化NSFW分类器"""
        # 制定本地模型路径
        self.nsfw_class = nsfw_class
        self.classifier = NudeDetector(model_path=model_path, inference_resolution=640,providers=providers)

        self.unsafe_threshold = unsafe_threshold  # NSFW判定阈值
        
    def classify_image(self, image_path):
        """对单张图片进行分类
        
        Args:
            image_path: 图片路径
            
        Returns:
            bool: 是否为不适合内容
            float: NSFW分数
        """
        try:
            # 使用NudeNet进行分类
            # nsfw_class =["FEMALE_GENITALIA_COVERED",
            #             "BUTTOCKS_EXPOSED",
            #             "FEMALE_BREAST_EXPOSED",
            #             "FEMALE_GENITALIA_EXPOSED",
            #             "ANUS_EXPOSED",
            #             "FEET_EXPOSED",                        "ARMPITS_EXPOSED",
            #             "ANUS_COVERED",
            #             "FEMALE_BREAST_COVERED",
            #             "BUTTOCKS_COVERED",
            #              ]
            # print(normalize_path(image_path))
            img = cv2.imdecode(
                np.fromfile(image_path, dtype=np.uint8), 
                cv2.IMREAD_COLOR
            )
               
            result = self.classifier.detect(img)
            if not result:
                print("result is None")
                return False, 0.0,""
            # print("len(result):",len(result))  
            score_total = 0
            score = 0
            class_name = ""
            # 找出相同class score最大的
            class_max_score = {}
            for re in result:
                if re['class'] in default_nsfw_class:
                    if re['class'] not in class_max_score:
                        class_max_score[re['class']] = re['score']
                    else:
                        if class_max_score[re['class']] < re['score']:
                            class_max_score[re['class']] = re['score']
            for cls,cls_score in class_max_score.items():   
                #求平方
                score_total = float(np.exp((cls_score-self.unsafe_threshold))*cls_score+score_total)
                if cls_score > score:
                    score = cls_score
                    class_name = cls
                # score_total = float(np.exp(score_total))
            # 获取分类结果
            # score = result[image_path]['unsafe']
            is_nsfw = score_total >= self.unsafe_threshold
            return is_nsfw, score_total,class_name
            
        except Exception:
            return False, 0.0,class_name
            
    def scan_directory(self, directory, recursive=True):
        """扫描目录，返回不适合的图片
        
        Args:
            directory: 要扫描的目录
            recursive: 是否递归扫描子目录
            
        Returns:
            dict: 包含不适合图片的路径和分数
        """
        # 收集所有图片文件
        results = {}
        image_files = []
        if recursive:
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                image_files.extend(Path(directory).rglob(ext))
        else:
            for ext in ('*.jpg', '*.jpeg', '*.png'):
                image_files.extend(Path(directory).glob(ext))
        
        image_files = [str(p) for p in image_files]
        if not image_files:
            return {}
            
        # 批量分类
        # results = self.classifier.detect_batch(image_files)
        for image_path in image_files:
            is_nsfw, score,class_name = self.classify_image(image_path)
            if is_nsfw:
                if class_name not in results:
                    results[class_name] = []    
                results[class_name].append(image_path)
                print(image_path)
                print(score)
                print(class_name)
                print("--------------------------------")
                
        return results

def main():
    # 创建分类器实例
    classifier = NSFWClassifier()
    
    # 获取输入路径
    input_path = input("请输入图片或目录路径: ").strip()
    if not os.path.exists(input_path):
        return
    
    # 根据输入类型进行处理
    if os.path.isfile(input_path):
        # 处理单个文件
        is_nsfw, score,class_name = classifier.classify_image(input_path)
        return {"不适合内容": [input_path]} if is_nsfw else {}
    else:
        # 处理目录
        results = classifier.scan_directory(input_path)
        return results

if __name__ == "__main__":
    main() 