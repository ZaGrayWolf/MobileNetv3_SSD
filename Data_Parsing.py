#How to parse VOC2012 DATASET
import os
import xml.etree.ElementTree as ET
import shutil

# Function to validate and organize your dataset
def prepare_voc_dataset(source_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'Annotations'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'JPEGImages'), exist_ok=True)
    
    # Copy and validate XML and image files
    for filename in os.listdir(source_dir):
        if filename.endswith('.xml'):
            # Validate XML
            try:
                ET.parse(os.path.join(source_dir, filename))
                shutil.copy(os.path.join(source_dir, filename), 
                            os.path.join(output_dir, 'Annotations', filename))
            except ET.ParseError:
                print(f"Invalid XML: {filename}")
        
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            shutil.copy(os.path.join(source_dir, filename), 
                        os.path.join(output_dir, 'JPEGImages', filename))
