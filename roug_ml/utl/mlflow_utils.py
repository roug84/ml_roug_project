from mlflow.tracking import MlflowClient
from beartype.typing import Tuple, List
import mlflow


def get_best_run(experiment_name: str, metric_key: str) -> Tuple[str, dict]:
    """
    Retrieves the best run and its parameters from a specified experiment.

    :param experiment_name: Name of the experiment
    :param metric_key: Key of the metric to use for determining the best run. The best run is
     determined by ordering the runs by this metric in descending order and picking the first one.
    :return: A tuple containing the ID of the best run and the parameters of the best run.
    :raises ValueError: If no such experiment exists.
    """
    client = MlflowClient()

    # Get the experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"No such experiment '{experiment_name}'")

    # Search for the best run in the experiment
    runs = client.search_runs([experiment.experiment_id], order_by=[f"metric.{metric_key} DESC"])

    # Assuming the first run is the best one
    best_run = runs[0]
    # retrieve the best run_id
    best_run_id = best_run.info.run_id
    # retrieve the best parameters
    best_params = best_run.data.params

    return best_run_id, best_params


def get_top_n_runs(experiment_name: str, metric_key: str, n: int) -> List[Tuple[str, dict]]:
    """
    Retrieves the top N runs and their parameters from a specified experiment.

    :param experiment_name: Name of the experiment.
    :param metric_key: Key of the metric to use for determining the top runs. The runs are
     ordered by this metric in descending order.
    :param n: Number of top runs to retrieve.
    :return: A list of tuples, each containing the ID of a run and the parameters of that run.
    :raises ValueError: If no such experiment exists or if N is less than or equal to 0.
    """
    if n <= 0:
        raise ValueError("The parameter 'n' should be greater than 0.")

    client = MlflowClient()

    # Get the experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise ValueError(f"No such experiment '{experiment_name}'")

    # Search for the best runs in the experiment
    runs = client.search_runs([experiment.experiment_id], order_by=[f"metric.{metric_key} DESC"], max_results=n)

    # Extract the run_ids and parameters for the top N runs
    top_n_runs = [(run.info.run_id, run.data.params) for run in runs]

    return top_n_runs


def get_or_create_experiment(name):
    experiment = mlflow.get_experiment_by_name(name)

    if experiment is None:
        # The experiment does not exist, create a new one
        experiment_id = mlflow.create_experiment(name)
    else:
        # The experiment exists, get its ID
        experiment_id = experiment.experiment_id

    return experiment_id


def load_top_models(top_n_runs: List[Tuple[str, dict]]) -> List:
    """
    Loads the models for the top N runs using MLflow.

    :param top_n_runs: A list of tuples, each containing the ID of a run and the parameters of that run.
    :return: A list of loaded models.
    """
    loaded_models = []
    for run in top_n_runs:
        run_id, _ = run  # We are only interested in the run ID for loading
        model = mlflow.sklearn.load_model("runs:/{}/pipeline".format(run_id))
        loaded_models.append(model)

    return loaded_models
