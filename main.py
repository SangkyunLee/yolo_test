from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx", imgsz=640, opset=12)


def main():
    print("Hello from yolo-test!")


if __name__ == "__main__":
    main()
