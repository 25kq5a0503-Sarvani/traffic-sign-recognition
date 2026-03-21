// ── Shared helpers ────────────────────────────────────────

function showAlert(id, msg, type) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = `alert alert-${type}`;
  el.textContent = msg;
  el.style.display = 'block';
}

function hideAlert(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = 'none';
}

// ── Voice Output (Text to Speech) ─────────────────────────

function speakResult(signName, confidence) {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();
    const msg = new SpeechSynthesisUtterance(
      `Traffic sign detected: ${signName}. Confidence: ${Math.round(confidence)} percent.`
    );
    msg.lang   = 'en-US';
    msg.rate   = 0.9;
    msg.volume = 1;
    window.speechSynthesis.speak(msg);
  }
}

// ── Upload drag-and-drop ──────────────────────────────────

function initUpload(boxId, inputId, previewId) {
  const box   = document.getElementById(boxId);
  const input = document.getElementById(inputId);
  const prev  = document.getElementById(previewId);
  if (!box || !input) return;

  box.addEventListener('dragover', e => {
    e.preventDefault();
    box.style.background = '#d0eeff';
  });
  box.addEventListener('dragleave', () => box.style.background = '');
  box.addEventListener('drop', e => {
    e.preventDefault();
    box.style.background = '';
    if (e.dataTransfer.files[0]) {
      input.files = e.dataTransfer.files;
      showPreview(e.dataTransfer.files[0], prev);
    }
  });

  input.addEventListener('change', () => {
    if (input.files[0]) showPreview(input.files[0], prev);
  });
}

function showPreview(file, imgEl) {
  if (!imgEl) return;
  const reader = new FileReader();
  reader.onload = e => {
    imgEl.src = e.target.result;
    imgEl.style.display = 'block';
  };
  reader.readAsDataURL(file);
}

// ── Predict page ──────────────────────────────────────────

function predictSign() {
  const input     = document.getElementById('img-input');
  const resultBox = document.getElementById('result-box');
  const signName  = document.getElementById('sign-name');
  const confEl    = document.getElementById('confidence');
  const top3Card  = document.getElementById('top3-card');
  const top3List  = document.getElementById('top3-list');

  if (!input || !input.files[0]) {
    showAlert('pred-alert', '⚠️ Please select an image first!', 'error');
    return;
  }

  hideAlert('pred-alert');
  showAlert('pred-alert', '🔍 Analyzing image...', 'info');
  if (resultBox) resultBox.style.display = 'none';
  if (top3Card)  top3Card.style.display  = 'none';

  const fd = new FormData();
  fd.append('file', input.files[0]);

  fetch('/api/predict', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(data => {
      hideAlert('pred-alert');
      if (data.error) {
        showAlert('pred-alert', '❌ ' + data.error, 'error');
        return;
      }

      // Main result
      signName.textContent    = data.sign_name;
      confEl.textContent      = `Confidence: ${data.confidence}%`;
      resultBox.style.display = 'block';

      // Top 3
      if (data.top3 && top3Card && data.confidence < 90) {
        top3List.innerHTML = data.top3.map((item, i) => `
          <div style="padding:10px 14px; margin:6px 0;
                      background:${i===0?'#e8f8f5':'#f8f9fa'};
                      border-left:4px solid ${i===0?'#02C39A':'#065A82'};
                      border-radius:6px; display:flex; justify-content:space-between">
            <span>${i===0?'🥇':i===1?'🥈':'🥉'} <b>${item.sign_name}</b></span>
            <span style="color:${i===0?'#02C39A':'#065A82'}; font-weight:bold">
              ${item.confidence}%
            </span>
          </div>
        `).join('');
        top3Card.style.display = 'block';
      }

      // 🔊 Voice Output
      speakResult(data.sign_name, data.confidence);
    })
    .catch(() => showAlert('pred-alert', '❌ Server error!', 'error'));
}

// ── Training page ─────────────────────────────────────────

function startTraining() {
  const btn      = document.getElementById('train-btn');
  const progWrap = document.getElementById('prog-wrap');
  const progBar  = document.getElementById('prog-bar');
  const progMsg  = document.getElementById('prog-msg');
  const graphImg = document.getElementById('graph-img');

  btn.disabled           = true;
  btn.textContent        = '⏳ Training...';
  progWrap.style.display = 'block';
  hideAlert('train-alert');
  if (graphImg) graphImg.style.display = 'none';

  fetch('/api/train', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      if (data.error) {
        showAlert('train-alert', data.error, 'error');
        btn.disabled    = false;
        btn.textContent = '🚀 Start Training';
        return;
      }
      pollProgress(btn, progBar, progMsg, graphImg);
    })
    .catch(() => {
      showAlert('train-alert', '❌ Server error. Is Flask running?', 'error');
      btn.disabled    = false;
      btn.textContent = '🚀 Start Training';
    });
}

function pollProgress(btn, progBar, progMsg, graphImg) {
  const interval = setInterval(() => {
    fetch('/api/train/progress')
      .then(r => r.json())
      .then(data => {
        progBar.style.width = data.percent + '%';
        progMsg.textContent = data.message;

        if (data.status === 'done') {
          clearInterval(interval);
          btn.disabled    = false;
          btn.textContent = '🚀 Start Training';
          const r = data.result;
          showAlert('train-alert',
            `✅ Training Complete! Accuracy: ${r.accuracy}% | Loss: ${r.loss} | Images: ${r.total_images}`,
            'success');
          if (graphImg) {
            graphImg.src           = '/static/images/training_graphs.png?t=' + Date.now();
            graphImg.style.display = 'block';
          }
        } else if (data.status === 'error') {
          clearInterval(interval);
          btn.disabled    = false;
          btn.textContent = '🚀 Start Training';
          showAlert('train-alert', '❌ Error: ' + data.message, 'error');
        }
      });
  }, 1500);
}

// ── Dataset page ──────────────────────────────────────────

function loadDatasetInfo() {
  const container = document.getElementById('dataset-stats');
  if (!container) return;

  fetch('/api/dataset/info')
    .then(r => r.json())
    .then(data => {
      const ti = document.getElementById('total-images');
      const nc = document.getElementById('num-classes');
      if (ti) ti.textContent = data.total_images.toLocaleString();
      if (nc) nc.textContent = data.num_classes;
    })
    .catch(() => {});
}

// ── Auto-init on load ─────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initUpload('upload-box', 'img-input', 'preview-img');
  loadDatasetInfo();

  const path = window.location.pathname;
  document.querySelectorAll('.navbar ul a').forEach(a => {
    if (a.getAttribute('href') === path) a.classList.add('active');
  });
});

// ── Webcam (Live page) ────────────────────────────────────

let webcamStream = null;

function startWebcam() {
  const video = document.getElementById('webcam');
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
      webcamStream = stream;
      video.srcObject = stream;
      video.style.display = 'block';
      document.getElementById('cam-placeholder').style.display = 'none';
      document.getElementById('start-btn').disabled  = true;
      document.getElementById('stop-btn').disabled   = false;
      document.getElementById('capture-btn').disabled = false;
      showAlert('cam-alert', '✅ Camera started! Point at a traffic sign!', 'success');
    })
    .catch(() => {
      showAlert('cam-alert', '❌ Camera access denied!', 'error');
    });
}

function stopWebcam() {
  if (webcamStream) {
    webcamStream.getTracks().forEach(t => t.stop());
    webcamStream = null;
    document.getElementById('webcam').srcObject = null;
    document.getElementById('webcam').style.display = 'none';
    document.getElementById('cam-placeholder').style.display = 'flex';
    document.getElementById('start-btn').disabled  = false;
    document.getElementById('stop-btn').disabled   = true;
    document.getElementById('capture-btn').disabled = true;
    showAlert('cam-alert', '⏹️ Camera stopped.', 'info');
  }
}

function captureAndPredict() {
  const video  = document.getElementById('webcam');
  const canvas = document.getElementById('canvas');

  if (!webcamStream) {
    showAlert('cam-alert', '❌ Please start camera first!', 'error');
    return;
  }

  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  showAlert('cam-alert', '🔍 Analyzing...', 'info');

  canvas.toBlob(blob => {
    const fd = new FormData();
    fd.append('file', blob, 'webcam_capture.png');

    fetch('/api/predict', { method: 'POST', body: fd })
      .then(r => r.json())
      .then(data => {
        if (data.error) {
          showAlert('cam-alert', '❌ ' + data.error, 'error');
          return;
        }
        document.getElementById('sign-name').textContent    = data.sign_name;
        document.getElementById('confidence').textContent   = `Confidence: ${data.confidence}%`;
        document.getElementById('result-box').style.display = 'block';
        showAlert('cam-alert', `✅ Detected: ${data.sign_name}!`, 'success');

        // 🔊 Voice Output
        speakResult(data.sign_name, data.confidence);
      })
      .catch(() => showAlert('cam-alert', '❌ Server error!', 'error'));
  }, 'image/png');
}