import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image

data = []
labels = []

# Check the correct path to the images folder
for folder in os.listdir("images"):  # Adjusted to "images"
    folder_path = os.path.join("images", folder)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = Image.open(img_path).resize((64, 64))
        img_array = np.array(img).flatten()
        data.append(img_array)
        labels.append(folder)

data = np.array(data)
labels = np.array(labels)

df = pd.DataFrame(data)
df['label'] = labels

# Displaying sample images from 'class1' and 'class2'
sample_class1 = df[df['label'] == 'class1'].iloc[0, :-1].values.reshape(64, 64, -1)
sample_class2 = df[df['label'] == 'class2'].iloc[0, :-1].values.reshape(64, 64, -1)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(sample_class1.astype(np.uint8))
plt.title("Class 1")
plt.subplot(1, 2, 2)
plt.imshow(sample_class2.astype(np.uint8))
plt.title("Class 2")
plt.show()
