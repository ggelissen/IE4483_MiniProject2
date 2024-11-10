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
test_dir = os.path.join('unbalanced', 'test')
output_file = 'unbalanced.csv'

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
        img = load_img(img_path, target_size=(32, 32))
        img_array = img_to_array(img) / 255.0
        test_images.append(img_array)
        image_ids.append(os.path.splitext(filename)[0]) 

test_images_array = np.array(test_images)
print("Shape of test_images_array:", test_images_array.shape)



#### Making predictions ####

predictions = model.predict(test_images_array)
predicted_classes = np.argmax(predictions, axis=1)


# Example of displaying the first 5 predictions
print("Predicted classes:", predicted_classes.flatten()[0])
#### Exporting predictions to CSV ####

output_df = pd.DataFrame({'id': image_ids, 'label': predicted_classes.flatten()})      # Create dataframe with image IDs and labels
output_df['id'] = output_df['id'].astype(int)
output_df.sort_values(by='id', inplace=True) 
output_df['label'] = output_df['label'].astype(int)           

output_df.to_csv(output_file, index=False)                                              # Export dataframe to CSV file

print(f'Predictions saved to {output_file}')

score=0
output_df.reset_index(drop=True, inplace=True)
print(output_df['label'][0])
for i in range(len(output_df['label'])):
    if output_df['label'][i] == int(i/50):
        score+=1
print('Accuracy:', 100*score/500,"%")