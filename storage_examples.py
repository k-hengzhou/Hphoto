"""
数据存储方案示例集合
适用于大量关键词-属性对的存储场景
包含多种存储方法的实现和对比
"""

import json
import zlib
import struct
import sqlite3
import pickle
import msgpack
from pathlib import Path
from typing import Dict, Any, Optional

class JsonStore:
    """JSON存储方案
    优点: 可读性好，使用简单
    缺点: 存储空间相对较大
    适用: 需要直接查看/编辑数据文件的场景
    """
    def __init__(self, file_path: str = 'data/keywords.json'):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
    def save(self, data: Dict) -> None:
        with open(self.file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self) -> Dict:
        if not self.file_path.exists():
            return {}
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

class MessagePackStore:
    """MessagePack存储方案
    优点: 存储空间小，读写速度快
    缺点: 文件不可直接读
    适用: 追求效率的生产环境
    """
    def __init__(self, file_path: str = 'data/keywords.msgpack'):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
    def save(self, data: Dict) -> None:
        with open(self.file_path, 'wb') as f:
            msgpack.pack(data, f)
    
    def load(self) -> Dict:
        if not self.file_path.exists():
            return {}
        with open(self.file_path, 'rb') as f:
            return msgpack.unpack(f)

class CompressedSqliteDB:
    """压缩的SQLite存储方案
    优点: 支持查询，存储空间小
    缺点: 需要数据库操作
    适用: 需要频繁查询的大数据集
    """
    def __init__(self, db_file: str = 'data/keywords.db'):
        self.db_file = Path(db_file)
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_file))
        self.cursor = self.conn.cursor()
        
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS keywords (
            keyword TEXT PRIMARY KEY,
            compressed_data BLOB
        )
        ''')
        
    def add_keyword(self, keyword: str, attributes: Dict) -> None:
        json_data = json.dumps(attributes, ensure_ascii=False)
        compressed = zlib.compress(json_data.encode())
        
        self.cursor.execute(
            'INSERT OR REPLACE INTO keywords VALUES (?, ?)',
            (keyword, compressed)
        )
        self.conn.commit()
        
    def get_keyword(self, keyword: str) -> Optional[Dict]:
        self.cursor.execute(
            'SELECT compressed_data FROM keywords WHERE keyword = ?',
            (keyword,)
        )
        result = self.cursor.fetchone()
        if result:
            decompressed = zlib.decompress(result[0])
            return json.loads(decompressed.decode())
        return None
    
    def close(self) -> None:
        self.conn.close()

class BinaryStore:
    """二进制存储方案
    优点: 存储空间最小
    缺点: 不易维护
    适用: 追求最小存储空间的场景
    """
    def __init__(self, file_path: str = 'data/keywords.bin'):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
    def save(self, data: Dict) -> None:
        with open(self.file_path, 'wb') as f:
            pickled = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            compressed = zlib.compress(pickled)
            f.write(struct.pack('!I', len(compressed)))
            f.write(compressed)
    
    def load(self) -> Dict:
        if not self.file_path.exists():
            return {}
        with open(self.file_path, 'rb') as f:
            size = struct.unpack('!I', f.read(4))[0]
            compressed = f.read(size)
            pickled = zlib.decompress(compressed)
            return pickle.loads(pickled)

def storage_comparison_example():
    """存储方案对比示例"""
    # 测试数据
    test_data = {
        "关键词1": {
            "属性1": "值1",
            "属性2": [1, 2, 3],
            "属性3": {"子属性": "值"}
        },
        "关键词2": {
            "属性1": "值2",
            "属性2": [4, 5, 6],
            "属性3": {"子属性": "值"}
        }
    }
    
    # 1. JSON存储
    json_store = JsonStore()
    json_store.save(test_data)
    
    # 2. MessagePack存储
    msgpack_store = MessagePackStore()
    msgpack_store.save(test_data)
    
    # 3. 压缩SQLite存储
    sqlite_store = CompressedSqliteDB()
    for keyword, attrs in test_data.items():
        sqlite_store.add_keyword(keyword, attrs)
    sqlite_store.close()
    
    # 4. 二进制存储
    binary_store = BinaryStore()
    binary_store.save(test_data)
    
    # 打印文件大小对比
    def get_file_size(path: Path) -> int:
        return path.stat().st_size if path.exists() else 0
    
    print("\n存储空间对比:")
    print(f"JSON: {get_file_size(json_store.file_path)} bytes")
    print(f"MessagePack: {get_file_size(msgpack_store.file_path)} bytes")
    print(f"SQLite: {get_file_size(sqlite_store.db_file)} bytes")
    print(f"Binary: {get_file_size(binary_store.file_path)} bytes")

if __name__ == '__main__':
    # 运行对比示例
    storage_comparison_example() 