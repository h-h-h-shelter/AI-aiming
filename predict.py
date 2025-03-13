from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("last.pt")

# Define path to video file
source = "test/test.mp4"

# Run inference on the source
results = model(source, stream=True)  # generator of Results objects

# Process results generator
cnt=0
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    # result.show()  # display to screen
    filename = f"test/result/{cnt}.jpg"
    result.save(filename=filename)  # save to disk
    cnt+=1