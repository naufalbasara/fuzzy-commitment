# ===== UTILITIES =====
import logging, cv2, numpy as np, tensorflow as tf

def face_detection(img, xml_path) -> tuple[int,int,int,int]:
    """Detect faces using haarcascade classifier from OpenCV, returning coordinates for bounding box"""
    face_cascade = cv2.CascadeClassifier(xml_path)
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=5)
    for (x,y,w,h) in faces:
        return x,y,w,h

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def get_vector(image_path, classifier, face_detection, xml_path) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    
    try:
        x,y,w,h = face_detection(img, xml_path)
        cropped = img[y-75:y+h+25, x-75:x+w+25]
    except:
        print(f"Face not found in {image_path}")
        cropped = img

    resized = cv2.resize(cropped, (224, 224), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f'captured.jpg', resized)
    preprocessed = np.expand_dims(resized, 0)
    user_vector = classifier.predict(preprocessed, verbose=0)

    return user_vector

def hamming_distance(a: np.ndarray, b: np.ndarray):
    return np.sum(a != b)