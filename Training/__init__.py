import os
from abc import abstractmethod,  ABC
import joblib
from sklearn.model_selection import train_test_split

class DataSetTrainer(ABC):
    '''
    Abstract base class for all training methods.
    '''

    saved_model_path = None

    def __init__(self, X, y, test_size=0.2, random_state=42):
        '''
        Initialize with the dataset.
        '''
        self.xTrain, self.xTest, self.yTrain, self.yTest = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state, 
            stratify=y
        )
        self.model = None

        

    @abstractmethod
    def train(self):
        """
        Method to train the model. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Method to evaluate the model performance.
        """
        pass

    def save_model(self, working_directory_path, model_file_name):
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        if not os.path.exists(f"{working_directory_path}/models"):
            os.makedirs(f"{working_directory_path}/models")

        self.saved_model_path = os.path.join(f"{working_directory_path}/models", f'{model_file_name}.joblib')
        
        joblib.dump(self.model, self.saved_model_path)
        print(f"Model saved to {self.saved_model_path}")   