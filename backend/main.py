import cv2
import numpy as np
import base64
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
TFLITE_MODEL_PATH = "model.tflite"
model_tflite = None

if os.path.exists(TFLITE_MODEL_PATH):
    try:
        import tflite_runtime.interpreter as tflite
        model_tflite = tflite.Interpreter(model_path=TFLITE_MODEL_PATH)
        model_tflite.allocate_tensors()
        print("TFLite model loaded successfully.")
    except ImportError:
        try:
            import tensorflow as tf
            model_tflite = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
            model_tflite.allocate_tensors()
            print("TFLite model loaded successfully using TensorFlow.")
        except Exception as e:
            print(f"Error loading TFLite model: {e}")
    except Exception as e:
        print(f"Error loading TFLite model: {e}")

def predict_emotion(face_image):
    face_resized = cv2.resize(face_image, (48, 48))
    img = np.array(face_resized).reshape(1, 48, 48, 1).astype('float32') / 255.0

    if model_tflite:
        input_details = model_tflite.get_input_details()
        output_details = model_tflite.get_output_details()
        model_tflite.set_tensor(input_details[0]['index'], img)
        model_tflite.invoke()
        pred = model_tflite.get_tensor(output_details[0]['index'])
        return labels[np.argmax(pred)]
    
    return "neutral"


haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'happy', 2: 'neutral', 3: 'sad'}

def extract_features(image):
    feature = np.array(image).reshape(1, 48, 48, 1)
    return feature / 255.0

@app.websocket("/ws/detect")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # The data is expected to be a base64 encoded jpeg image 
            # e.g., "data:image/jpeg;base64,/9j/4AAQSkZJRgABA..."
            if data.startswith("data:image"):
                base64_data = data.split(",")[1]
            else:
                base64_data = data
            
            try:
                img_data = base64.b64decode(base64_data)
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            except Exception as decode_error:
                await websocket.send_json({"error": "Failed to decode image"})
                continue
            
            if frame is None:
                await websocket.send_json({"error": "Invalid frame"})
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 3)
            
            result = {"emotion": "neutral", "faces": []}
            
            if len(faces) > 0:
                # Process the largest face
                faces_sorted = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                (x, y, w, h) = faces_sorted[0]
                face = gray[y:y+h, x:x+w]
                face_resized = cv2.resize(face, (48, 48))
                
                emotion = predict_emotion(face)
                result["emotion"] = emotion

                
                result["faces"] = [{"x": int(x), "y": int(y), "w": int(w), "h": int(h)}]
                
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket Error: {e}")
