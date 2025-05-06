from Training import DataSetTrainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import numpy as np

class RandomForestTrainer(DataSetTrainer):

    """
    Implementation for Random Forest training.
    """
    def __init__(self, dataset, target, oversample=False, random_state=42):

        X = dataset.drop(columns=[target])
        y = dataset[target]
        
        X = np.array(X)
        y = np.array(y) 
        if oversample:
            smote = SMOTE(random_state=random_state)
            X_resampled, y_resampled = smote.fit_resample(X, y)

            super().__init__(X_resampled, y_resampled, random_state = random_state)
        else:
            # Train-test split in super class
            super().__init__(X, y, random_state = random_state)

    def train(self):
        """
        Train a Random Forest classifier.
        """
        if self.xTrain is None or self.yTrain is None:
            error_msg = "Training data not initialized. Call __init__() with correct parameters first."
            raise ValueError(error_msg)

        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight='balanced',
            max_depth=4
        )
        self.model.fit(self.xTrain, self.yTrain)
        return self.model
    
    def evaluate(self):
        """
        Evaluate the Random Forest model.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        if self.xTest is None or self.yTest is None:
            error_msg = "Test data not initialized. Call __init__() with correct parameters first."
            raise ValueError(error_msg)
        
        yPred = self.model.predict(self.xTest)
        
        # Calculate metrics
        conf_matrix = confusion_matrix(self.yTest, yPred)
        class_report = classification_report(self.yTest, yPred)
        accuracy = accuracy_score(self.yTest, yPred)
      
        return {
            'confusion_matrix': conf_matrix,
            'accuracy': accuracy,
            'classification_report': class_report
        }