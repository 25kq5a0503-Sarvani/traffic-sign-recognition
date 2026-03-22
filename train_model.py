import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from PIL import Image
import os, json

# ── Constants ──────────────────────────────────────────────
IMG_SIZE    = 32
NUM_CLASSES = 43
BATCH_SIZE  = 32
EPOCHS      = 15

# 43 Traffic sign names
SIGN_NAMES = {
    0:'Speed limit (20km/h)',     1:'Speed limit (30km/h)',
    2:'Speed limit (50km/h)',     3:'Speed limit (60km/h)',
    4:'Speed limit (70km/h)',     5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)',    9:'No passing',
    10:'No passing (over 3.5t)', 11:'Right-of-way at intersection',
    12:'Priority road',          13:'Yield',
    14:'Stop',                   15:'No vehicles',
    16:'No trucks',              17:'No entry',
    18:'General caution',        19:'Dangerous curve left',
    20:'Dangerous curve right',  21:'Double curve',
    22:'Bumpy road',             23:'Slippery road',
    24:'Road narrows right',     25:'Road work',
    26:'Traffic signals',        27:'Pedestrians',
    28:'Children crossing',      29:'Bicycles crossing',
    30:'Beware ice/snow',        31:'Wild animals crossing',
    32:'End of all speed limits',33:'Turn right ahead',
    34:'Turn left ahead',        35:'Ahead only',
    36:'Go straight or right',   37:'Go straight or left',
    38:'Keep right',             39:'Keep left',
    40:'Roundabout mandatory',   41:'End of no passing',
    42:'End no passing (3.5t)'
}

def load_data(data_path):
    """loading images & labels from dataset folder"""
    images, labels = [], []
    for class_id in range(NUM_CLASSES):
        class_path = os.path.join(data_path, str(class_id))
        if not os.path.exists(class_path):
            continue
        for img_name in os.listdir(class_path):
            if not img_name.lower().endswith(('.png','.jpg','.jpeg','.ppm')):
                continue
            try:
                img = Image.open(os.path.join(class_path, img_name)).convert('RGB')
                img = img.resize((IMG_SIZE, IMG_SIZE))
                images.append(np.array(img))
                labels.append(class_id)
            except Exception:
                pass
    return np.array(images), np.array(labels)

def build_model():
    """Building the CNN model"""
    model = Sequential([
        Conv2D(32,(3,3), activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3)),
        Conv2D(32,(3,3), activation='relu'),
        MaxPooling2D(2,2),
        Dropout(0.25),
        Conv2D(64,(3,3), activation='relu'),
        Conv2D(64,(3,3), activation='relu'),
        MaxPooling2D(2,2),
        Dropout(0.25),
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train(data_path='dataset/Train', progress_callback=None):
    """Full training pipeline"""
    if progress_callback: progress_callback("Loading dataset...", 5)

    X, y = load_data(data_path)
    if len(X) == 0:
        raise ValueError("No images found! Check dataset/Train folder.")

    X = X / 255.0
    y_cat = to_categorical(y, NUM_CLASSES)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42)

    if progress_callback: progress_callback(f"Loaded {len(X)} images. Training...", 25)

    model = build_model()
    history = model.fit(X_train, y_train,
                        batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_split=0.2, verbose=1)

    if progress_callback: progress_callback("Saving model & graphs...", 88)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    os.makedirs('model', exist_ok=True)
    model.save('model/traffic_model.h5')
    _save_graphs(history)

    if progress_callback: progress_callback("Training complete!", 100)

    return {'accuracy': round(acc*100,2), 'loss': round(loss,4),
            'epochs': EPOCHS, 'total_images': len(X)}

def _save_graphs(history):
    fig, axes = plt.subplots(1, 2, figsize=(12,4))
    axes[0].plot(history.history['accuracy'],     label='Train', color='#065A82')
    axes[0].plot(history.history['val_accuracy'], label='Val',   color='#02C39A')
    axes[0].set_title('Model Accuracy'); axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy');      axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history['loss'],     label='Train', color='#F96167')
    axes[1].plot(history.history['val_loss'], label='Val',   color='#E8A838')
    axes[1].set_title('Model Loss'); axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss');      axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs('static/images', exist_ok=True)
    plt.savefig('static/images/training_graphs.png', dpi=120, bbox_inches='tight')
    plt.close()

def predict(img_path):
    """If we give Image It Predicts the size"""
    from tensorflow.keras.models import load_model
    if not os.path.exists('model/traffic_model.h5'):
        raise FileNotFoundError("Model not found! Train first.")
    model = load_model('model/traffic_model.h5')
    img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(np.array(img)/255.0, axis=0)
    preds    = model.predict(arr)
    
    # Top 3 predictions
    top3_idx  = np.argsort(preds[0])[::-1][:3]
    top3 = [
        {
            'class_id':   int(i),
            'sign_name':  SIGN_NAMES.get(int(i), 'Unknown'),
            'confidence': round(float(preds[0][i])*100, 2)
        }
        for i in top3_idx
    ]
    
    return {
        'class_id':   int(top3_idx[0]),
        'sign_name':  top3[0]['sign_name'],
        'confidence': top3[0]['confidence'],
        'top3':       top3
    }
