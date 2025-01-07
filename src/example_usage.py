# 导入必要的库
from face_organizer import FaceOrganizer    # 导入人脸识别器类
import argparse                             # 用于处理命令行参数
import os                                   # 用于处理文件路径
import sys
import locale

# 设置默认编码
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='人脸识别照片整理工具')
    
    # 添加命令行参数
    parser.add_argument('--input', '-i', 
                       type=str, 
                       default='photos',
                       help='输入文件夹路径，默认为 "photos"')
    
    parser.add_argument('--output', '-o',
                       type=str,
                       default='organized_photos',
                       help='输出文件夹路径，默认为 "organized_photos"')
    
    parser.add_argument('--faces-db', '-f',
                       type=str,
                       default='known_faces.pkl',
                       help='人脸数据库文件路径，默认为 "known_faces.pkl"')
    
    parser.add_argument('--threshold', '-t',
                       type=float,
                       default=0.5,
                       help='人脸匹配阈值 (0-1)，默认为 0.5')
    
    parser.add_argument('--no-gui', '-n',
                       action='store_true',
                       help='使用无GUI模式，只处理可识别的人脸')
    
    parser.add_argument('--clean-db', '-c',
                       action='store_true',
                       help='清理人脸数据库，对每个人的特征向量进行聚类')
    
    parser.add_argument('--no-update-db', '-u',
                       action='store_true',
                       help='不更新人脸数据库')
    
    parser.add_argument('--no-backup-db', '-b',
                       action='store_true',
                       help='不创建数据库备份')
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 检查输入目录
    # if not check_input_directory(args.input):
    #     return 1
    
    # # 确保输出目录存在
    # try:
    #     ensure_directory(args.output)
    # except OSError as e:
    #     print(str(e))
    #     return 1
    
    try:
        # 创建人脸识别器实例
        organizer = FaceOrganizer(
            faces_db_path=args.faces_db,
            threshold=args.threshold,
            update_db=not args.no_update_db,
            backup_db=not args.no_backup_db
        )
        
        # 处理文件夹
        total_files, processed_files = organizer.process_directory(
            input_dir=args.input,      # 输入路径
            output_dir=args.output,    # 输出路径
            gui_mode=not args.no_gui   # GUI模式设置
        )
        
        # 打印处理结果
        print("\n处理结果统计：")
        print(f"总文件数: {total_files}")
        print(f"成功处理: {processed_files}")
        print(f"处理失败: {total_files - processed_files}")
        
        if args.clean_db:
            organizer.clean_face_database()
        
        return 0
        
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        return 1

# 程序入口
if __name__ == "__main__":
    exit(main()) 