import mlflow
from roug_ml.utl.mlflow_utils import get_or_create_experiment


class MLPipeline:
    def __init__(self, in_mlflow_experiment_name) -> None:
        self.mlflow_experiment_id = None
        self.mlflow_experiment_name = in_mlflow_experiment_name

    def set_mlflow_params(self):
        """
        Sets the tracking URI for mlflow and initializes the mlflow experiment
        """
        mlflow.set_tracking_uri("http://localhost:8000")
        self.mlflow_experiment_id = get_or_create_experiment(self.mlflow_experiment_name)

    def collect_data(self):
        """
        The first step in any machine learning pipeline is data collection. This may involve
        gathering data from various sources like databases, files, APIs, web scraping, or even
        creating synthetic data.
        """
        pass

    def preprocess_data(self):
        """
        Data Preprocessing and Cleaning: Once data has been collected, it needs to be preprocessed
        and cleaned. This can involve dealing with missing values, handling outliers, correcting
        inconsistent data formats, etc.
        """
        pass

    def extract_features(self):
        """
        Feature Engineering and Selection: In this step, new features are created from the existing
        data which can help improve the model's performance. Feature selection is also done in this
        stage to choose the most relevant features to train the model.
        """
        pass

    def split_data(self):
        """
        Data Splitting: The dataset is usually split into a training set, validation set (optional),
        and a test set. The training set is used to train the model, the validation set is used to
        fine-tune the model parameters, and the test set is used to evaluate the final model.
        """
        pass

    def model_training(self):
        """
        Model Training: In this step, different machine learning algorithms are applied to the
        training data. The choice of algorithm depends on the nature of the problem (e.g.,
        classification, regression), the data, and the business context.
        """
        pass

    def evaluation(self):
        """
        Model Evaluation and Selection: After training, models are evaluated using suitable metrics
        (accuracy, precision, recall, F1 score, ROC AUC, etc., depending on the task). The best
        performing model is then selected.
        """
        pass

    def hyperoptimize(self):
        """
        Model Optimization and Hyperparameter Tuning: The selected model is further optimized to
        improve its performance. This is usually done by tuning its hyperparameters. Methods such
        as Grid Search, Random Search, or Bayesian Optimization are used for this purpose.
        """
        pass

    def validate(self):
        """
        Model Validation: The final model is validated using the test set which was not used in the
        training or model selection process.
        """
        pass

    def deploy(self):
        """
        Model Deployment: After validation, the model is deployed in the real-world environment.
        This could be a server, a cloud-based platform, or directly embedded into an application.
        """
        pass

    def maintenance(self):
        """
        Model Monitoring and Maintenance: After deployment, the model's performance is monitored
        over time. If the model's performance degrades, it may need to be retrained or replaced.
        """
        pass

