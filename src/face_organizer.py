import os
import cv2
import numpy as np
import insightface
from PIL import Image
import pickle
from datetime import datetime
import tkinter as tk
from tkinter import ttk, simpledialog
from tkinter import messagebox
from tqdm import tqdm  # 在文件开头添加导入

# 人脸识别和组织核心类
class FaceOrganizer:
    def __init__(self, model_path=None, faces_db_path='known_faces.pkl', 
                 threshold=0.5, update_db=True, backup_db=True):
        """初始化人脸识别器
        
        Args:
            model_path: 人脸识别模型路径
            faces_db_path: 人脸数据库路径
            threshold: 人脸匹配阈值(0-1)
            update_db: 是否更新数据库
            backup_db: 是否备份数据库
        """
        self.threshold = threshold
        self.faces_db_path = faces_db_path
        self.update_db = update_db
        self.backup_db = backup_db
        self.known_faces = self._load_faces_db()
        
        # 初始化人脸分析器
        try:
            self.face_analyzer = insightface.app.FaceAnalysis(
                model_name='buffalo_l',
                model_path=model_path,
                providers=['CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            raise Exception(f"模型加载失败: {str(e)}")

    def _load_faces_db(self):
        """加载人脸数据库
        
        Returns:
            dict: 人脸数据库字典，key为人名，value为特征向量
        """
        if os.path.exists(self.faces_db_path):
            try:
                with open(self.faces_db_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                return {}
        return {}

    def _save_faces_db(self):
        """保存人脸数据库"""
        if not self.update_db:
            return
            
        try:
            # 创建备份
            if self.backup_db and os.path.exists(self.faces_db_path):
                backup_path = f"{self.faces_db_path}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.bak"
                os.rename(self.faces_db_path, backup_path)
            
            # 保存数据库
            with open(self.faces_db_path, 'wb') as f:
                db_to_save = {}
                for name, faces in self.known_faces.items():
                    db_to_save[name] = []
                    for face in faces:
                        db_to_save[name].append(
                            face.astype(np.float32) if not isinstance(face, np.ndarray) else face
                        )
                pickle.dump(db_to_save, f)
                
        except Exception as e:
            pass

    def detect_faces(self, image_path):
        """检测图片中的人脸
        
        Args:
            image_path: 图片路径
            
        Returns:
            tuple: (faces, img) 人脸列表和原图
        """
        try:
            img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return None
                
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = self.face_analyzer.get(img)
            # 过滤掉检测得分小于0.7的矩形框
            faces = [face for face in faces if face.det_score > 0.5]
            # 过滤掉特征向量长度小于512的矩形框
            faces = [face for face in faces if len(face.embedding) >= 512 ]
            return (faces, img) if faces else (None, img)
            
        except Exception:
            return None

    def compare_face(self,image_path):
        """比较人脸"""
        result = self.detect_faces(image_path)
        if result is None:
            return False
        faces, img = result
        max_similarity = 0
        matched_name = None
        print("len(faces):",len(faces)) 
        for face in faces:
            embedding = face.embedding
            for known_name, known_embedding in self.known_faces.items():    
                if isinstance(known_embedding, list):
                    print("known_embedding:",len(known_embedding))
                    similarity = max(self._calculate_cosine_similarity(embedding, ke) 
                                   for ke in known_embedding)
                else:   
                    print("known_embedding 1:",len(known_embedding))
                    similarity = self._calculate_cosine_similarity(embedding, known_embedding)
                print("known_name:",known_name,"similarity:",similarity)
                if similarity > max_similarity:
                    max_similarity = similarity
                    if similarity > self.threshold:
                        matched_name = known_name
        return matched_name
    def register_face(self, face_embedding, person_name):
        """注册新人脸"""
        if person_name and person_name.strip():
            # 将face_embedding转换为numpy数组
            embedding = face_embedding.embedding.astype(np.float32)
            
            if person_name not in self.known_faces:
                self.known_faces[person_name] = []
                self.known_faces[person_name].append(embedding)
            else:
                self.known_faces[person_name].append(embedding)

            self._save_faces_db()
            print(f"已将此人脸注册为: {person_name}")
            return person_name
        return None
    # def register_face(self,image_path, person_name):
    #     """注册新人脸"""
    #     if person_name and person_name.strip():
    #         face_embedding = self.detect_faces([image_path])
    #         if face_embedding is None:
    #             print(f"无法检测到人脸: {image_path}")
    #             return None
    #         if isinstance(self.known_faces[person_name], list):
    #             self.known_faces[person_name].append(face_embedding)
    #         else:
    #             self.known_faces[person_name] = [self.known_faces[person_name], face_embedding]
    #         self._save_faces_db()
    #         print(f"已将此人脸注册为: {person_name}")
    #         return person_name
    #     return None
    def get_face_db(self):
        return self.known_faces
    def _handle_unknown_face(self, image_path, face_embedding):
        """处理未识别的人脸，提供按钮界面"""
        print(f"\n在 {image_path} 中发现未识别的人脸")
        
        # 读取并显示图片
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"无法读取图片: {image_path}")
            return None
        
        # 获取人脸位置并绘制矩形框
        faces = self.face_analyzer.get(image)
        if len(faces) > 0:
            face = faces[0]
            bbox = face.bbox.astype(int)
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(image, "未识别的人脸", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # 创建并设置图片窗口
        window_name = "未识别的人脸"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # 计算图片显示大小（保持宽高比，最大宽度为800）
        height, width = image.shape[:2]
        max_width = 800
        if width > max_width:
            scale = max_width / width
            display_width = int(width * scale)
            display_height = int(height * scale)
        else:
            display_width = width
            display_height = height
        
        # 设置窗口大小和位置（左上角）
        cv2.resizeWindow(window_name, display_width, display_height)
        cv2.moveWindow(window_name, 0, 0)
        cv2.imshow(window_name, image)
        
        # 创建按钮窗口
        root = tk.Tk()
        root.title("操作选择")
        
        # 获取屏幕尺寸
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # 设置窗口大小
        window_width = 300
        window_height = 400
        
        # 计算窗口位置（屏幕中央）
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        
        # 设置窗口大小和位置
        root.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        # 设置窗口始终置顶
        root.attributes('-topmost', True)
        
        # 存储用户选择
        result = {'action': None, 'name': None}
        
        def register_face():
            """注册新人脸"""
            name = simpledialog.askstring("输入", "请输入这个人的名字:", parent=root)
            if name and name.strip():
                result['action'] = 'register'
                result['name'] = name.strip()
                root.destroy()
        
        def skip():
            """跳过当前图片"""
            result['action'] = 'skip'
            root.destroy()
        
        def save():
            """保存当前图片并可选注册人脸"""
            # 创建保存对话框
            save_dialog = tk.Toplevel(root)
            save_dialog.title("保存选项")
            save_dialog.geometry("400x200")
            
            # 创建变量存储用户输入
            name_var = tk.StringVar()
            register_var = tk.BooleanVar(value=False)
            
            # 创建输入框和标签
            ttk.Label(save_dialog, text="请输入人名:").pack(pady=5)
            name_entry = ttk.Entry(save_dialog, textvariable=name_var)
            name_entry.pack(pady=5)
            
            # 创建复选框
            ttk.Checkbutton(save_dialog, text="同时注册人脸特征", 
                           variable=register_var).pack(pady=10)
            
            def confirm():
                name = name_var.get().strip()
                if name:
                    base_name = name.split('_')[0]  # 获取基础名字（不含重复计数）
                    print("base_name:",base_name)
                    # 如果选择注册且人名已存在
                    if register_var.get():
                        if base_name in self.known_faces:
                            # 计算当前重复次数
                            count = 1
                            while f"{base_name}_{count}" in self.known_faces:
                                count += 1
                            register_name = f"{base_name}_{count}"
                            print("register_name:",register_name)
                            self.known_faces[register_name] = face_embedding
                            self._save_faces_db()
                            print(f"已将此人脸注册为: {register_name}")
                        else:
                            # 新名字直接注册
                            self.known_faces[base_name] = face_embedding
                            self._save_faces_db()
                            print(f"已将此人脸注册为: {base_name}")
                    
                    result['action'] = 'save'
                    result['name'] = base_name  # 保存时使用基础名字
                    save_dialog.destroy()
                    root.destroy()
                else:
                    tk.messagebox.showerror("错误", "人名不能为空！", parent=save_dialog)
            
            def cancel():
                save_dialog.destroy()
            
            # 创建按钮
            btn_frame = ttk.Frame(save_dialog)
            btn_frame.pack(pady=20)
            ttk.Button(btn_frame, text="确定", command=confirm).pack(side=tk.LEFT, padx=10)
            ttk.Button(btn_frame, text="取消", command=cancel).pack(side=tk.LEFT, padx=10)
            
            # 设置模态对话框
            save_dialog.transient(root)
            save_dialog.grab_set()
            name_entry.focus()
        
        def quit_program():
            """退出程序"""
            result['action'] = 'quit'
            root.destroy()
        
        # 创建按钮样式
        style = ttk.Style()
        style.configure('Custom.TButton', padding=10, font=('微软雅黑', 12))
        
        # 创建按钮框架
        btn_frame = ttk.Frame(root, padding="20")
        btn_frame.pack(fill=tk.BOTH, expand=True)
        
        # 添加按钮
        ttk.Button(btn_frame, text="注册新人脸 (Y)", 
                   command=register_face, style='Custom.TButton').pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="跳过 (N)", 
                   command=skip, style='Custom.TButton').pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="保存图片 (S)", 
                   command=save, style='Custom.TButton').pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="退出程序 (Q)", 
                   command=quit_program, style='Custom.TButton').pack(fill=tk.X, pady=10)
        
        # 添加键盘快捷键
        def on_key(event):
            if event.char == 'y':
                register_face()
            elif event.char == 'n':
                skip()
            elif event.char == 's':
                save()
            elif event.char == 'q':
                quit_program()
        
        root.bind('<Key>', on_key)
        
        # 运行界面
        root.mainloop()
        cv2.destroyWindow(window_name)
        
        # 处理结果
        if result['action'] == 'register' and result['name']:
            self.known_faces[result['name']] = face_embedding
            self._save_faces_db()
            print(f"已将此人脸注册为: {result['name']}")
            return result['name']
        elif result['action'] == 'save' and result['name']:
            return result['name']
        elif result['action'] == 'quit':
            print("用户选择退出程序")
            exit(0)
        
        return None

    def process_directory(self, input_dir, output_dir, gui_mode=True):
        """处理整个文件夹"""
        print(f"开始处理文件夹: {input_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 首先统计总文件数
        total_files = 0
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    total_files += 1
        
        if total_files == 0:
            print("未找到需要处理的图片文件")
            return 0, 0
        
        # 创建进度条
        processed_files = 0
        pbar = tqdm(total=total_files, 
                    desc="处理进度", 
                    unit="张",
                    ncols=100,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        
        try:
            # 处理文件
            for root, _, files in os.walk(input_dir):
                for file in files:
                    print("files num:",len(files)-processed_files)
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                        image_path = os.path.join(root, file)
                        print("set_description:")
                        # 更新进度条描述
                        pbar.set_description(f"处理: {os.path.basename(image_path)}")
                        print("process_image:") 
                        # 处理图片
                        if self.process_image(image_path, output_dir, gui_mode):
                            processed_files += 1
                        
                        # 更新进度条
                        pbar.update(1)
        
        finally:
            # 确保进度条正确关闭
            pbar.close()
        
        # 打印最终统计信息
        print(f"\n处理完成！共处理 {total_files} 张图片，成功分类 {processed_files} 张")
        print(f"成功率: {processed_files/total_files*100:.1f}%")
        
        return total_files, processed_files

    def _calculate_cosine_similarity(self, embedding1, embedding2):
        """计算两个特征向量的余弦相似度"""
        # 计算点积
        dot_product = np.dot(embedding1, embedding2)
        # 计算向量的模
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        # 计算余弦相似度
        return dot_product / (norm1 * norm2)
    # 检测输入的图片是否在人脸库中
    def is_face_in_db(self, image_path):
        """检测输入的图片是否在人脸库中"""
        result = self.detect_faces(image_path)
        if not result:
            return False
        faces, img = result
        for face in faces:
            embedding = face.embedding
            for known_name, known_embedding in self.known_faces.items():
                if isinstance(known_embedding, list):
                    # 如果人脸库中的人脸是多张图片的平均值，则计算最大相似度
                    similarity = max(self._calculate_cosine_similarity(embedding, ke) 
                                   for ke in known_embedding)   
                else:
                    # 如果人脸库中的人脸是单张图片，则计算相似度
                    similarity = self._calculate_cosine_similarity(embedding, known_embedding)
                if similarity > self.threshold:
                    return True
            return False    
    def process_image(self, image_path, output_dir, gui_mode=True):
        """处理单张图片"""
        # 检测人脸
        print("image_path:",image_path)
        result = self.detect_faces(image_path)
        print("result:")
        if not result:
            return False
        
        faces, img = result
        processed = False
        
        # 创建未识别人脸目录
        unknown_dir = os.path.join(output_dir, "未识别的人脸")
        os.makedirs(unknown_dir, exist_ok=True)
        print("image_path:",image_path)
        for face in faces:
            # 获取人脸特征
            embedding = face.embedding
            print("len(embedding):",len(embedding))
            # 查找相似人脸
            name = None
            max_similarity = 0
            
            for known_name, known_embedding in self.known_faces.items():
                if isinstance(known_embedding, list):
                    print("known_name:",known_name,"len(known_embedding):",len(known_embedding))
                    similarity = max(self._calculate_cosine_similarity(embedding, ke) 
                                   for ke in known_embedding)
                    print("known_name:",known_name,"similarity:",similarity)
                else:
                    similarity = self._calculate_cosine_similarity(embedding, known_embedding)
                    print("known_name:",known_name,"similarity:",similarity)
                if similarity > max_similarity:
                    max_similarity = similarity
                    
                    if similarity > self.threshold:
                        name = known_name
            print("name:",name) 
            if name is None and gui_mode:
                # 未找到匹配的人脸，尝试注册新人脸
                name = self._handle_unknown_face(image_path, embedding)
                
                # 如果用户选择跳过，将图片保存到未识别人脸文件夹
                if name is None:
                    filename = os.path.basename(image_path)
                    base, ext = os.path.splitext(filename)
                    unknown_path = os.path.join(unknown_dir, f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}")
                    
                    try:
                        # 将RGB转回BGR并编码为字节
                        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                        _, img_encoded = cv2.imencode(ext, img_bgr)
                        # 直接写入字节数据
                        with open(unknown_path, 'wb') as f:
                            f.write(img_encoded.tobytes())
                        print(f"已将未识别的人脸保存至: {unknown_path}")
                        processed = True
                    except Exception as e:
                        print(f"保存未识别人脸失败 {unknown_path}: {str(e)}")
                    continue
            if name is None and not gui_mode:
                return False
            
            if name:
                # 获取基础人名（去掉_后的数字）
                base_name = name.split('_')[0]
                
                # 创建对应的输出目录（使用基础人名）
                person_dir = os.path.join(output_dir, base_name)
                os.makedirs(person_dir, exist_ok=True)
                
                # 生成输出文件名
                filename = os.path.basename(image_path)
                base, ext = os.path.splitext(filename)
                output_path = os.path.join(person_dir, f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}")
                
                # 保存图片
                try:
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    _, img_encoded = cv2.imencode(ext, img_bgr)
                    with open(output_path, 'wb') as f:
                        f.write(img_encoded.tobytes())
                    print(f"已保存图片到 {base_name} 文件夹: {output_path}")
                    processed = True
                except Exception as e:
                    print(f"保存图片失败 {output_path}: {str(e)}")
        
        return processed