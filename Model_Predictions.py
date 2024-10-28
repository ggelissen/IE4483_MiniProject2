######

# IE4483 Artificial Intelligence and Data Mining
# Mini Project 2 - Dogs/Cats Binary Classification using CNN

######


#### Import statements ####

import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


#### Loading model and input/output files ####

# Load model from directory
model = load_model('cnn_model.keras')

# Define test folder and output file
test_dir = 'datasets/test'
output_file = 'submission.csv'


#### Image Processing ####

# Create test object and assign test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224), 
    batch_size=32,              
    class_mode=None,         
    shuffle=False           
)

# Extract image filenames and IDs
image_filenames = test_generator.filenames
image_ids = [os.path.splitext(os.path.basename(fname))[0] for fname in image_filenames]


#### Making predictions ####

predictions = model.predict(test_generator)             # Predict class of each test sample {0-1}
binary_predictions = (predictions > 0.5).astype(int)    # Convert to binary {0,1}


#### Exporting predictions to CSV ####

output_df = pd.DataFrame({'id': image_ids, 'label': binary_predictions[:, 0]})   # Create dataframe with image IDs and labels
output_df['label'] = output_df['label'].apply(lambda x: 1 if x == 1 else 0)      # Add results into dataframe
output_df.to_csv(output_file, index=False)                                       # Export dataframe to CSV file