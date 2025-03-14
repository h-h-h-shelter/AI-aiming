from ultralytics import YOLO

model = YOLO("runs\\train\\train4\\weights\\last.pt")


if __name__ == "__main__":
    results = model.train(data= "datasets/data.yaml", epoch=300, batch=-1, imgsz=640, save=True, project="runs/train",patience=600)
