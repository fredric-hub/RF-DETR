from fastapi import FastAPI, UploadFile, File
from rfdetr import RFDETRBase
from PIL import Image
import io
import supervision as sv

app = FastAPI()

# Initialize the model (Weights are downloaded on first run)
# Using 'RFDETRBase' or 'RFDETRSmall' as used in the notebook
# RF-DETR-Base is a good balance for T4
model = RFDETRBase() 

@app.get("/")
def health_check():
    return {"status": "running", "model": "RF-DETR"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Read image
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    
    # Run Inference
    # The notebook typically uses a confidence threshold of 0.5
    detections = model.predict(image, threshold=0.5)
    
    # Format results for response
    results = []
    # supervision Detections object contains arrays
    for box, class_id, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
        results.append({
            "box": box.tolist(),
            "class_id": int(class_id),
            "confidence": float(confidence),
            # Optional: Map class_id to name if you have the list
        })
        
    return {"detections": results}