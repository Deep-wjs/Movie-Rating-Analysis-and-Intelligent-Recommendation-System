import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ORLFaceDataLoader:
    """Load and preprocess ORL face dataset"""
    
    def __init__(self, data_path='orl_faces', test_size=0.2):
        """
        Initialize data loader
        
        Args:
            data_path: Path to ORL dataset folder (s1-s40)
            test_size: Proportion of test set (20% as per requirement)
        """
        self.data_path = data_path
        self.test_size = test_size
        self.images = []
        self.labels = []
        self.label_names = []
        
    def load_data(self):
        """
        Load images from s1-s40 folders
        
        Returns:
            X_train, X_test, y_train, y_test: Training and testing data
        """
        print("Loading ORL face dataset...")
        
        # Get all subject folders (s1 to s40)
        subject_dirs = sorted([d for d in os.listdir(self.data_path) 
                              if os.path.isdir(os.path.join(self.data_path, d))])
        
        label = 0
        for subject_dir in subject_dirs:
            subject_path = os.path.join(self.data_path, subject_dir)
            self.label_names.append(subject_dir)
            
            # Load all images in this subject folder
            image_files = sorted([f for f in os.listdir(subject_path) 
                                 if f.endswith(('.pgm', '.jpg', '.png'))])
            
            for image_file in image_files:
                image_path = os.path.join(subject_path, image_file)
                try:
                    # Load image and convert to grayscale
                    img = Image.open(image_path).convert('L')
                    img_array = np.array(img).flatten()
                    
                    self.images.append(img_array)
                    self.labels.append(label)
                except Exception as e:
                    print(f"Error loading {image_path}: {e}")
            
            label += 1
        
        # Convert to numpy arrays
        X = np.array(self.images, dtype=np.float32)
        y = np.array(self.labels)
        
        print(f"Loaded {len(X)} images from {len(subject_dirs)} subjects")
        print(f"Image shape (flattened): {X.shape}")
        
        # Split into train and test sets (80-20 split)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} images")
        print(f"Testing set: {len(X_test)} images")
        
        return X_train, X_test, y_train, y_test
    
    def get_image_shape(self):
        """Get the original image shape"""
        if len(self.images) > 0:
            # Assuming all images are same size, calculate from flattened size
            flattened_size = len(self.images[0])
            # ORL images are typically 112x92 pixels
            return (112, 92)
        return None
