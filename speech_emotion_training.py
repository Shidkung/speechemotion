import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
from tqdm import tqdm

class EmotionRecognitionSystem:
    """
    A class for building a speech emotion recognition system using MFCC features.
    Supports training on .flac audio files with emotion labels from a JSON file.
    Adapted for studio-based dataset structure.
    """
    
    def __init__(self, n_mfcc=13, sample_rate=22050):
        """Initialize the emotion recognition system with parameters for MFCC extraction."""
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.classifier = None
        self.scaler = StandardScaler()
        
    def extract_mfcc(self, audio_path):
        """
        Extract MFCC features from an audio file.
        
        Parameters:
        -----------
        audio_path : str
            Path to the audio file (.flac)
            
        Returns:
        --------
        mfcc_features : ndarray
            MFCC features
        """
        # Load audio file
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract MFCCs
        mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        
        # Add delta and delta-delta features to capture dynamics
        delta_mfcc = librosa.feature.delta(mfcc_features)
        delta2_mfcc = librosa.feature.delta(mfcc_features, order=2)
        
        # Stack all features
        combined_features = np.vstack([mfcc_features, delta_mfcc, delta2_mfcc])
        
        # Calculate statistics over time (mean and std)
        feature_vector = np.hstack([
            np.mean(combined_features, axis=1),
            np.std(combined_features, axis=1)
        ])
        
        return feature_vector
    
    def load_dataset(self, base_dir, label_file):
        """
        Load and prepare dataset from a structured directory of audio files and a JSON label file.
        Adapted for a studio-based dataset structure.
        
        Parameters:
        -----------
        base_dir : str
            Base directory containing studio folders (studio001-studio010)
        label_file : str
            Path to the JSON file containing emotion labels
            
        Returns:
        --------
        X : ndarray
            MFCC features
        y : ndarray
            Emotion labels
        """
        # Load label data
        with open(label_file, 'r') as f:
            labels = json.load(f)
        
        X = []
        y = []
        file_info = []  # To store metadata about each file
        
        print(f"Loading dataset from {base_dir}...")
        
        # Get list of studio directories
        studio_dirs = [d for d in os.listdir(base_dir) if d.startswith('studio') and os.path.isdir(os.path.join(base_dir, d))]
        
        for studio_dir in sorted(studio_dirs):
            studio_path = os.path.join(base_dir, studio_dir)
            
            # Get recording types (clip, con, middle)
            recording_types = [d for d in os.listdir(studio_path) if os.path.isdir(os.path.join(studio_path, d))]
            
            for rec_type in recording_types:
                rec_type_path = os.path.join(studio_path, rec_type)
                
                # Process all .flac files in this recording type directory
                flac_files = [f for f in os.listdir(rec_type_path) if f.endswith('.flac')]
                
                # Use tqdm for progress bar
                for file in tqdm(flac_files, desc=f"Processing {studio_dir}/{rec_type}"):
                    file_path = os.path.join(rec_type_path, file)
                    
                    # Check if this file has a label
                    if file in labels:
                        try:
                            # Get emotion label
                            emotion = labels[file][0]['assigned_emo']
                            
                            # Extract MFCC features
                            mfcc_features = self.extract_mfcc(file_path)
                            
                            X.append(mfcc_features)
                            y.append(emotion)
                            
                            # Store metadata
                            file_info.append({
                                'file_name': file,
                                'studio': studio_dir,
                                'recording_type': rec_type,
                                'emotion': emotion,
                                'path': file_path
                            })
                        except Exception as e:
                            print(f"Error processing {file}: {e}")
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Create DataFrame for easier analysis
        data_df = pd.DataFrame(file_info)
        
        # Data analysis
        self._analyze_dataset(data_df)
        
        return X, y, data_df
    
    def _analyze_dataset(self, data_df):
        """
        Analyze the loaded dataset and print statistics.
        
        Parameters:
        -----------
        data_df : DataFrame
            DataFrame containing metadata about the audio files
        """
        print("\n===== Dataset Analysis =====")
        
        # Total samples
        print(f"Total samples: {len(data_df)}")
        
        # Emotion distribution
        print("\nEmotion distribution:")
        emotion_counts = data_df['emotion'].value_counts()
        for emotion, count in emotion_counts.items():
            print(f"  {emotion}: {count} samples ({count/len(data_df)*100:.2f}%)")
        
        # Show distribution by recording type
        print("\nSamples by recording type:")
        type_counts = data_df['recording_type'].value_counts()
        for rec_type, count in type_counts.items():
            print(f"  {rec_type}: {count} samples ({count/len(data_df)*100:.2f}%)")
        
        # Show distribution by studio
        print("\nSamples by studio:")
        studio_counts = data_df['studio'].value_counts()
        for studio, count in studio_counts.items():
            print(f"  {studio}: {count} samples ({count/len(data_df)*100:.2f}%)")
        
        # Cross-tabulation of emotion by recording type
        print("\nEmotion distribution by recording type:")
        emotion_by_type = pd.crosstab(data_df['emotion'], data_df['recording_type'], normalize='columns') * 100
        print(emotion_by_type.round(2))
        
        print("\n============================")
    
    def train(self, X, y, test_size=0.2):
        """
        Train a classifier on MFCC features for emotion recognition.
        
        Parameters:
        -----------
        X : ndarray
            MFCC features
        y : ndarray
            Emotion labels
        test_size : float
            Proportion of the dataset to include in the test split
            
        Returns:
        --------
        results : dict
            Dictionary containing model evaluation results
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Normalize the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train the classifier
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=1000,
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate_init=0.001,
            random_state=42,
            verbose=True
        )
        
        print("Training the model...")
        self.classifier.fit(X_train_scaled, y_train)
        
        # Evaluate the classifier
        y_pred = self.classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nModel accuracy: {accuracy:.4f}")
        
        # Print detailed classification report
        print("\nClassification Report:")
        class_report = classification_report(y_test, y_pred, output_dict=True)
        print(classification_report(y_test, y_pred))
        
        # Plot confusion matrix
        cm, cm_normalized = self.plot_confusion_matrix(y_test, y_pred)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'confusion_matrix_normalized': cm_normalized,
            'y_true': y_test,
            'y_pred': y_pred
        }
        
        return results
    
    def train_with_cross_validation(self, X, y, n_splits=5):
        """
        Train the model using cross-validation.
        
        Parameters:
        -----------
        X : ndarray
            MFCC features
        y : ndarray
            Emotion labels
        n_splits : int
            Number of folds for cross-validation
            
        Returns:
        --------
        results : dict
            Dictionary containing cross-validation results
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, classification_report
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Store results for each fold
        fold_results = []
        
        # Perform cross-validation
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            print(f"\n===== Fold {fold+1}/{n_splits} =====")
            
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create and train classifier
            clf = MLPClassifier(
                hidden_layer_sizes=(256, 128, 64),
                max_iter=500,  # Reduced max_iter for faster CV
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size=32,
                learning_rate_init=0.001,
                random_state=42,
                verbose=False
            )
            
            # Train model
            clf.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = clf.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Fold {fold+1} accuracy: {accuracy:.4f}")
            
            # Store fold results
            fold_results.append({
                'fold': fold+1,
                'accuracy': accuracy,
                'classification_report': classification_report(y_test, y_pred, output_dict=True),
                'y_true': y_test,
                'y_pred': y_pred
            })
        
        # Calculate average metrics
        avg_accuracy = np.mean([r['accuracy'] for r in fold_results])
        print(f"\nAverage accuracy across {n_splits} folds: {avg_accuracy:.4f}")
        
        # Train final model on all data
        X_scaled = self.scaler.fit_transform(X)
        
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=1000,
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=32,
            learning_rate_init=0.001,
            random_state=42,
            verbose=True
        )
        
        print(f"\nTraining final model on all data...")
        self.classifier.fit(X_scaled, y)
        
        return {
            'avg_accuracy': avg_accuracy,
            'fold_results': fold_results,
            'final_model': self.classifier
        }
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """
        Plot a confusion matrix for the emotion classification results.
        
        Parameters:
        -----------
        y_true : ndarray
            True emotion labels
        y_pred : ndarray
            Predicted emotion labels
            
        Returns:
        --------
        cm : ndarray
            Confusion matrix
        cm_normalized : ndarray
            Normalized confusion matrix
        """
        # Get unique emotion labels
        labels = sorted(list(set(y_true)))
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=labels,
                    yticklabels=labels)
        plt.title('Normalized Confusion Matrix')
        plt.ylabel('True Emotion')
        plt.xlabel('Predicted Emotion')
        plt.tight_layout()
        plt.show()
        
        # Plot raw counts
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels,
                    yticklabels=labels)
        plt.title('Confusion Matrix (Counts)')
        plt.ylabel('True Emotion')
        plt.xlabel('Predicted Emotion')
        plt.tight_layout()
        plt.show()
        
        return cm, cm_normalized
    
    def predict_emotion(self, audio_path):
        """
        Predict the emotion from an audio file.
        
        Parameters:
        -----------
        audio_path : str
            Path to the audio file
            
        Returns:
        --------
        emotion : str
            Predicted emotion
        probabilities : dict
            Probability for each emotion class
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained yet. Please train the model first.")
        
        # Extract MFCC features
        mfcc_features = self.extract_mfcc(audio_path)
        
        # Scale features
        mfcc_features_scaled = self.scaler.transform(mfcc_features.reshape(1, -1))
        
        # Make prediction
        emotion = self.classifier.predict(mfcc_features_scaled)[0]
        probabilities = self.classifier.predict_proba(mfcc_features_scaled)[0]
        
        # Get emotion classes
        emotion_classes = self.classifier.classes_
        
        # Create probabilities dictionary
        prob_dict = {emotion_classes[i]: probabilities[i] for i in range(len(emotion_classes))}
        
        return emotion, prob_dict
    
    def predict_batch(self, data_df, audio_dir=None):
        """
        Make predictions for a batch of audio files.
        
        Parameters:
        -----------
        data_df : DataFrame
            DataFrame containing file paths
        audio_dir : str, optional
            Base directory for audio files (if paths in data_df are relative)
            
        Returns:
        --------
        results_df : DataFrame
            DataFrame with original data plus predictions
        """
        if self.classifier is None:
            raise ValueError("Classifier not trained yet. Please train the model first.")
        
        # Create copy of input DataFrame
        results_df = data_df.copy()
        
        # Add columns for predictions
        results_df['predicted_emotion'] = None
        results_df['confidence'] = None
        
        # Add columns for probabilities of each emotion
        for emotion in self.classifier.classes_:
            results_df[f'prob_{emotion}'] = None
        
        # Process each file
        for idx, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Predicting"):
            # Get audio path
            if 'path' in row:
                audio_path = row['path']
            else:
                # Construct path from components
                if audio_dir is None:
                    raise ValueError("audio_dir must be provided if paths not in data_df")
                file_name = row['file_name']
                studio = row['studio']
                rec_type = row['recording_type']
                audio_path = os.path.join(audio_dir, studio, rec_type, file_name)
            
            # Make prediction
            try:
                emotion, prob_dict = self.predict_emotion(audio_path)
                
                # Store prediction and confidence
                results_df.at[idx, 'predicted_emotion'] = emotion
                results_df.at[idx, 'confidence'] = prob_dict[emotion]
                
                # Store all probabilities
                for emo, prob in prob_dict.items():
                    results_df.at[idx, f'prob_{emo}'] = prob
            except Exception as e:
                print(f"Error predicting {audio_path}: {e}")
        
        return results_df
    
    def visualize_features(self, audio_path, title=None):
        """
        Visualize audio waveform, MFCC features, and more for an audio file.
        
        Parameters:
        -----------
        audio_path : str
            Path to the audio file
        title : str, optional
            Title for the plot
        """
        # Load audio
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract MFCC features (not aggregated for visualization)
        mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        
        # Create subplots
        fig, axs = plt.subplots(3, 1, figsize=(15, 10))
        
        # Plot waveform
        librosa.display.waveshow(y, sr=sr, ax=axs[0])
        axs[0].set_title('Waveform')
        
        # Plot MFCC features
        img = librosa.display.specshow(mfcc_features, x_axis='time', ax=axs[1])
        axs[1].set_title('MFCC Features')
        fig.colorbar(img, ax=axs[1], format='%+2.0f dB')
        
        # Plot mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img2 = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=sr, fmax=8000, ax=axs[2])
        axs[2].set_title('Mel Spectrogram')
        fig.colorbar(img2, ax=axs[2], format='%+2.0f dB')
        
        # Add main title if provided
        if title:
            fig.suptitle(title, fontsize=16)
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, model_path):
        """Save the trained model and scaler to a file."""
        if self.classifier is None:
            raise ValueError("No trained model to save")
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'n_mfcc': self.n_mfcc,
            'sample_rate': self.sample_rate
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a trained model and scaler from a file."""
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.n_mfcc = model_data.get('n_mfcc', self.n_mfcc)
        self.sample_rate = model_data.get('sample_rate', self.sample_rate)
        print(f"Model loaded from {model_path}")
    
    def visualize_comparing_emotions(self, data_df, emotions_to_compare=None, n_samples=3):
        """
        Visualize and compare MFCC features for different emotions.
        
        Parameters:
        -----------
        data_df : DataFrame
            DataFrame containing file metadata
        emotions_to_compare : list, optional
            List of emotions to compare (defaults to all emotions in data_df)
        n_samples : int
            Number of samples to show for each emotion
        """
        if emotions_to_compare is None:
            emotions_to_compare = sorted(data_df['emotion'].unique())
        
        # Create a figure with rows for emotions and columns for samples
        fig, axs = plt.subplots(len(emotions_to_compare), n_samples, 
                                figsize=(5*n_samples, 4*len(emotions_to_compare)))
        
        for i, emotion in enumerate(emotions_to_compare):
            # Get samples for this emotion
            emotion_samples = data_df[data_df['emotion'] == emotion].sample(n_samples)
            
            for j, (_, sample) in enumerate(emotion_samples.iterrows()):
                # Load audio
                y, sr = librosa.load(sample['path'], sr=self.sample_rate)
                
                # Extract MFCC features
                mfcc_features = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
                
                # Plot MFCC
                img = librosa.display.specshow(mfcc_features, x_axis='time', ax=axs[i, j])
                
                # Set title
                if j == 0:
                    axs[i, j].set_ylabel(emotion)
                
                file_parts = os.path.basename(sample['path']).split('_')
                axs[i, j].set_title(f"{file_parts[0]}_{file_parts[2]}")
        
        # Add colorbar
        fig.colorbar(img, ax=axs, format='%+2.0f')
        
        plt.suptitle('Comparing MFCC Features Across Emotions', fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize the emotion recognition system
    ser_system = EmotionRecognitionSystem(n_mfcc=20)
    
    # Set paths to your dataset
    base_dir = "Dataset"  # Directory containing studio001, studio002, etc.
    label_file = "emotion_label.json"
    
    # Load dataset
    X, y, data_df = ser_system.load_dataset(base_dir, label_file)
    
    # Option 1: Train with simple train/test split
    results = ser_system.train(X, y)
    
    # Option 2: Train with cross-validation for more robust evaluation
    # cv_results = ser_system.train_with_cross_validation(X, y, n_splits=5)
    
    # Save the model
    ser_system.save_model('emo_rec_model.pkl')
    
    # Visualize features for different emotions
    ser_system.visualize_comparing_emotions(data_df, n_samples=3)
    
    # Test the model on specific examples
    test_file = data_df.sample(1).iloc[0]['path']
    emotion, probabilities = ser_system.predict_emotion(test_file)
    
    print(f"\nPredicted emotion for test file: {emotion}")
    print("Probabilities:")
    for emo, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
        print(f"  {emo}: {prob:.4f}")
    
    # Visualize features for the test file
    ser_system.visualize_features(test_file, title=f"Features for {emotion} emotion")