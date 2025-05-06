from abc import abstractmethod,  ABC
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

    def save_model(self, working_directory_path):
        
        pass