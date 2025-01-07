import os
import sys
import shutil
import subprocess
from pathlib import Path

def create_spec_file(main_file: str, app_name: str) -> str:
    """创建 spec 文件"""
    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['{main_file}'],
    pathex=['src'],
    binaries=[],
    datas=[
        ('assets/*.png', 'icon'),
        ('config/*.json', 'config'),
    ],
    hiddenimports=[
        'insightface',
        'onnxruntime',
        'PyQt5',
        'src',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='{app_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icons/app.ico' if os.path.exists('icons/app.ico') else None
)
'''
    spec_file = f'{app_name}.spec'
    with open(spec_file, 'w', encoding='utf-8') as f:
        f.write(spec_content)
    return spec_file

def build_exe(main_file: str = 'main.py', app_name: str = 'PhotoManager'):
    """构建 exe 文件"""
    try:
        # 1. 检查环境
        print("检查 PyInstaller...")
        try:
            import PyInstaller
        except ImportError:
            print("正在安装 PyInstaller...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyinstaller'])

        # 2. 清理旧的构建文件
        print("清理旧的构建文件...")
        for dir_name in ['build', 'dist']:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)

        # 3. 创建 spec 文件
        print("创建 spec 文件...")
        spec_file = create_spec_file(main_file, app_name)

        # 4. 执行构建
        print("开始构建...")
        subprocess.check_call(['pyinstaller', spec_file, '--clean'])

        # 5. 复制额外文件
        print("复制额外文件...")
        dist_dir = Path('dist') / app_name
        if not dist_dir.exists():
            dist_dir = Path('dist')  # 单文件模式下的路径

        # 复制配置文件
        if os.path.exists('config'):
            shutil.copytree('config', dist_dir / 'config', dirs_exist_ok=True)

        # 复制图标文件
        if os.path.exists('icons'):
            shutil.copytree('icons', dist_dir / 'icons', dirs_exist_ok=True)

        print(f"\n构建成功！exe 文件位于: {dist_dir}")
        
        # 6. 创建启动脚本
        with open(dist_dir / 'run.bat', 'w') as f:
            f.write(f'@echo off\nstart {app_name}.exe')

    except Exception as e:
        print(f"构建失败: {str(e)}")
        return False

    return True

if __name__ == '__main__':
    # 可以通过命令行参数指定主文件和应用名称
    main_file = sys.argv[1] if len(sys.argv) > 1 else 'main.py'
    app_name = sys.argv[2] if len(sys.argv) > 2 else 'PhotoManager'
    
    print(f"开始构建 {app_name}...")
    if build_exe(main_file, app_name):
        print("构建完成！")
    else:
        print("构建失败！") 