# Face Photo Organizer

A tool for organizing photos based on face recognition.

[中文](README.md) | [日本語](README_JP.md)

## Features

- Face recognition and photo organization
- Support for Chinese/Japanese/English paths
- Duplicate photo detection
- NSFW content filtering
- Face clustering for unknown faces
- Modern GUI interface
- Multi-threaded processing

## Requirements

- Python 3.8+
- OpenCV
- InsightFace
- PyQt5
- Other dependencies listed in requirements.txt

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-photo-organizer.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download required models:
   ```bash
   python src/check_env.py
   ```

## Usage

Run the main program:   
```bash
python Hphoto.py
```

### Basic Operations

1. **Add Photos**: Click "Add Folder" or "Add File" to import photos
2. **Register Faces**: Right-click on a photo to register new faces
3. **Organize Photos**: Photos will be automatically organized by person
4. **Clean Database**: Use the cleaning tool to optimize the face database

### Advanced Features

- **Face Clustering**: Automatically group unknown faces
- **Duplicate Detection**: Find and manage duplicate photos
- **NSFW Filtering**: Filter inappropriate content
- **Database Management**: Clean and optimize face database

## Configuration

Edit `config/config.json` to customize:

- Face recognition threshold
- Clustering parameters
- NSFW detection settings
- Database backup options

## License

MIT License

## Acknowledgments

- InsightFace for face recognition
- NudeNet for NSFW detection
- PyQt5 for GUI framework
