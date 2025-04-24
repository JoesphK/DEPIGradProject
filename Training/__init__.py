from abc import abstractmethod,  ABC

class DataSetTrainer(ABC):
    '''
    Abstract base class for all training methods.
    '''

    saved_model_path = None

    def __init__(self, dataset):
        '''
        Initialize with the dataset.
        '''
        super().__init__(dataset)

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