from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os, threading, json
import train_model as tm

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

training_progress = {'status': 'idle', 'message': '', 'percent': 0, 'result': None}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/live')
def live_page():
    return render_template('live.html')

@app.route('/live-auto')
def live_auto_page():
    return render_template('live_auto.html')

@app.route('/dataset')
def dataset():
    return render_template('dataset.html')

@app.route('/train')
def train_page():
    return render_template('train.html')
    
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/train', methods=['POST'])
def start_training():
    global training_progress
    if training_progress['status'] == 'running':
        return jsonify({'error': 'Training already in progress!'}), 400
    training_progress = {'status': 'running', 'message': 'Starting...', 'percent': 0, 'result': None}

    def run_training():
        global training_progress
        try:
            def callback(msg, pct):
                training_progress['message'] = msg
                training_progress['percent'] = pct
            result = tm.train(data_path='dataset/Train', progress_callback=callback)
            training_progress['status']  = 'done'
            training_progress['result']  = result
            training_progress['percent'] = 100
        except Exception as e:
            training_progress['status']  = 'error'
            training_progress['message'] = str(e)

    threading.Thread(target=run_training, daemon=True).start()
    return jsonify({'message': 'Training started!'})

@app.route('/api/train/progress')
def training_progress_api():
    return jsonify(training_progress)

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    allowed = {'png','jpg','jpeg','ppm','bmp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({'error': f'File type .{ext} not allowed'}), 400
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    try:
        result = tm.predict(filepath)
        result['image_url'] = f'/static/uploads/{filename}'
        return jsonify(result)
    except FileNotFoundError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/api/dataset/info')
def dataset_info():
    train_path = 'dataset/Train'
    info = {'total_images': 0, 'num_classes': 0, 'classes': []}
    if os.path.exists(train_path):
        for cid in range(tm.NUM_CLASSES):
            cp = os.path.join(train_path, str(cid))
            if os.path.exists(cp):
                count = len([f for f in os.listdir(cp)
                              if f.lower().endswith(('.png','.jpg','.ppm'))])
                info['classes'].append({'id': cid, 'name': tm.SIGN_NAMES[cid], 'count': count})
                info['total_images'] += count
                info['num_classes']  += 1
    return jsonify(info)

if __name__ == '__main__':
    print("🚦 Traffic Sign Recognition App Starting...")
    print("   Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True)
