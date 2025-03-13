from ultralytics import YOLO

model = YOLO("last.pt")
if __name__ == "__main__":
    results = model.train(data= "datasets/data.yaml", time=3, batch=-1, imgsz=640, save=True, project="runs/train")
