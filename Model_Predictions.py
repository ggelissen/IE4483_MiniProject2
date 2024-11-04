######

# IE4483 Artificial Intelligence and Data Mining
# Mini Project 2 - Dogs/Cats Binary Classification using CNN

######


#### Import statements ####

import os
import numpy as np
import pandas as pd
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


#### Loading model and input/output files ####

# Load model and input/output files
model = load_model('cnn_model.keras')
test_dir = os.path.join('datasets', 'test')
output_file = 'submission.csv'

# List all images in the test directory
test_images = [f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
test_df = pd.DataFrame({'filename': test_images})


#### Image Processing ####

# Prepare list to hold image data and filenames
test_images = []
image_ids = []

# Load and preprocess images
for filename in os.listdir(test_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  
        img_path = os.path.join(test_dir, filename)
        img = load_img(img_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        test_images.append(img_array)
        image_ids.append(os.path.splitext(filename)[0]) 

test_images_array = np.array(test_images)


#### Making predictions ####

predictions = model.predict(test_images_array)              # Predict class of each test sample {0-1}
binary_predictions = (predictions > 0.5).astype(int)        # Convert to binary {0,1}


#### Exporting predictions to CSV ####

output_df = pd.DataFrame({'id': image_ids, 'label': binary_predictions.flatten()})      # Create dataframe with image IDs and labels
output_df['id'] = output_df['id'].astype(int)
output_df.sort_values(by='id', inplace=True) 
output_df['label'] = output_df['label'].apply(lambda x: 1 if x == 1 else 0)           

output_df.to_csv(output_file, index=False)                                              # Export dataframe to CSV file

print(f'Predictions saved to {output_file}')