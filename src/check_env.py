import importlib.metadata
import sys

def check_requirements():
    """检查环境依赖"""
    try:
        # 使用UTF-8编码读取requirements.txt
        with open('requirements.txt', 'r', encoding='utf-8') as f:
            # 过滤掉注释和空行
            requirements = []
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    # 只保留包名和版本号
                    pkg_info = line.split('#')[0].strip()
                    if pkg_info:
                        requirements.append(pkg_info)
    
        missing = []
        for requirement in requirements:
            try:
                # 解析包名和版本要求
                if '>=' in requirement:
                    pkg_name = requirement.split('>=')[0].strip()
                else:
                    pkg_name = requirement
                
                # 检查包是否已安装
                importlib.metadata.version(pkg_name)
            except importlib.metadata.PackageNotFoundError:
                missing.append(requirement)
            except Exception as e:
                print(f"检查 {requirement} 时出错: {str(e)}")
                
        if missing:
            print("\n缺少以下依赖:")
            for pkg in missing:
                print(f"  - {pkg}")
            print("\n请使用以下命令安装:")
            print("pip install -r requirements.txt")
            sys.exit(1)
        else:
            print("✅ 所有依赖已正确安装!")
            
    except Exception as e:
        print(f"检查依赖时出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    check_requirements() 