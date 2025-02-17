# 顔認識写真整理ツール

顔認識に基づいて写真を整理するツール。

[中文](README.md) | [English](README_EN.md)

## 特徴

- 顔認識と写真整理
- 日本語/中国語/英語のパスに対応
- 重複写真の検出
- NSFW（不適切なコンテンツ）のフィルタリング
- 未知の顔のクラスタリング
- モダンなGUIインターフェース
- マルチスレッド処理

## 必要条件

- Python 3.8以上
- OpenCV
- InsightFace
- PyQt5
- その他の依存関係はrequirements.txtに記載

## インストール

1. リポジトリをクローン：
   ```bash
   git clone https://github.com/your-username/face-photo-organizer.git
   ```
2. 依存関係をインストール：
   ```bash
   pip install -r requirements.txt
   ```
3. 必要なモデルをダウンロード：
   ```bash
   python src/check_env.py
   ```
  
## 使用方法

メインプログラムを実行：
```bash
python Hphoto.py
```


### 基本操作

1. **写真の追加**：「フォルダを追加」または「ファイルを追加」をクリック
2. **顔の登録**：写真を右クリックして新しい顔を登録
3. **写真の整理**：写真は人物ごとに自動的に整理
4. **データベースのクリーニング**：顔データベースを最適化

### 高度な機能

- **顔のクラスタリング**：未知の顔を自動的にグループ化
- **重複検出**：重複写真の検出と管理
- **NSFWフィルタリング**：不適切なコンテンツのフィルタリング
- **データベース管理**：顔データベースのクリーニングと最適化

## 設定

`config/config.json`を編集してカスタマイズ：

- 顔認識の閾値
- クラスタリングパラメータ
- NSFW検出設定
- データベースバックアップオプション

## ライセンス

MITライセンス

## 謝辞

- 顔認識：InsightFace
- NSFW検出：NudeNet
- GUIフレームワーク：PyQt5
