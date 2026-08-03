from ultralytics import YOLO



def main():
    model = YOLO("yolov8n.pt")
    model.export(format="onnx", imgsz=640, opset=12)

    print("yolo model exported!")


if __name__ == "__main__":
    main()
