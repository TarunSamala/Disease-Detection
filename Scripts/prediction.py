from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('./Model/tomato_disease_detection_model.h5')

# Function to preprocess a single image
def preprocess_image(image_path, target_size=(128, 128)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array[np.newaxis, ...]  # Add batch dimension

image_path = './PlantVillage/Tomato_Leaf_Mold/0a555f63-bf03-4958-8993-e1932b8dce9f___Crnl_L.Mold 9064.JPG'

input_image = preprocess_image(image_path)

prediction = model.predict(input_image)

predicted_class = "Mold" if prediction[0][0] > 0.5 else "Healthy"

plt.imshow(load_img(image_path))
plt.title(f'Predicted class: {predicted_class}')
plt.show()
