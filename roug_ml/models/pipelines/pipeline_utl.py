import os
from sklearn.pipeline import Pipeline
from roug_ml.utl.paths_utl import create_dir_path
from roug_ml.utl.etl.model_etl import save_default_model, load_default_model


class PipeModel(Pipeline):
    def __init__(self, steps, **params):
        """Instantiate the pipeline and the report.
        """
        super().__init__(steps, **params)
        self.report = {}

    def save(self, model_path, model_name):
        """Save each step of the pipeline.
        :param model_path: The path where to save it.
        :type model_path: string
        :param model_name: The name of the model.
        :type model_name: string
        """
        path_pipe = os.path.join(model_path, model_name)
        create_dir_path(path_pipe)

        for step_id, step in self.steps:

            if hasattr(step, 'save'):
                step.save(path_pipe, step_id)

            else:
                save_default_model(step, path_pipe, step_id)

    def load(self, model_path, model_name):
        """Load each step of the pipeline.
        :param model_path: The path where to save it.
        :type model_path: string
        :param model_name: The name of the model.
        :type model_name: string
        """
        path_pipe = os.path.join(model_path, model_name)

        new_steps = []
        for step_id, step in self.steps:

            if hasattr(step, 'load'):
                step_tmp = step.load(path_pipe, step_id)

            else:
                step_tmp = load_default_model(path_pipe, step_id)

            new_steps.append((step_id, step_tmp))

        self.steps = new_steps

        return self

    def get_pipeline_params(self):
        """Returns parameters for this pipeline that can be used to set parameters.
        :return: The parameters set for this pipeline.
        :rtype: dict
        """
        params = self.get_params()
        for step in params['steps']:
            params.pop(step[0])
        params.pop('steps')
        return params

    def get_report(self):
        """Return the reports from each step of the pipeline.
        """
        # Get report from each step
        for step_id, step in self.steps:
            if hasattr(step, 'report'):
                self.report[step_id] = step.report
            else:
                self.report[step_id] = None

        return self.report