<h1>🗑️ Garbage Classification Model</h1>

<p>This CNN model classifies garbage images into 7 categories: cardboard, glass, metal, paper, plastic, trash, and biodegradable. Built with TensorFlow and trained on a labeled dataset of ~2,500 images, it achieves ~92% accuracy.</p>

<h3>🔗 <a href="https://huggingface.co/vincenthuynh/GarbageMLModel_CS549" target="_blank">View on Hugging Face</a></h3>

<h2>📦 Usage</h2>
<pre><code>
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model('GarbageMLModel_CS549.h5')
img = Image.open("example.jpg").resize((224, 224))
img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

prediction = model.predict(img_array)
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash', 'biodegradable']
print("Predicted:", classes[np.argmax(prediction)])
</code></pre>

<h2>🧠 Features</h2>
<ul>
  <li>7 waste categories</li>
  <li>92% validation accuracy</li>
  <li>Grad-CAM support for model explainability</li>
</ul>

<p>Created by Vincent Huynh | 📧 vintendohuynh@gmail.com</p>
