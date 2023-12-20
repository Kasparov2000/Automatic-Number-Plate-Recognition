# ANPR System with YOLOv5

## Video Presentation

For a comprehensive overview and demonstration of the ANPR system, please watch the [ANPR System Video Presentation](https://drive.google.com/file/d/1CqWvu2isHOfhMvXQ2Ig3Rc4sncIBHSpS/view?usp=sharing).

## ANPR Web Application

Explore the ANPR system with YOLOv5 through the interactive [ANPR Web App](https://drive.google.com/file/d/1CqWvu2isHOfhMvXQ2Ig3Rc4sncIBHSpS/view?usp=sharing). The web app provides a user-friendly interface for license plate detection and recognition.

## YOLOv5 Integration

### Prerequisites

- Clone the YOLOv5 repository:

```bash
git clone https://github.com/ultralytics/yolov5
```

- Install YOLOv5 requirements:

```bash
pip install -r ./yolov5/requirements.txt
```

### Data Preparation for YOLOv5

1. Copy images and labels into YOLOv5 data folders:

```bash
python prepare_yolo_data.py
```

### Train YOLOv5 Model

Train the YOLOv5 model for license plate detection:

```bash
python ./yolov5/train.py --data data.yaml --cfg ./yolov5/models/yolov5s.yaml --batch-size 8 --name Model --epochs 100
```

Feel free to explore and enhance the ANPR system based on your requirements!