import os
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core import Workspace, Experiment
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.pipeline.core import Pipeline


def main():
    interactive_auth = InteractiveLoginAuthentication(tenant_id=os.getenv("TENANT_ID"))
    work_space = Workspace.from_config(auth=interactive_auth)
    prepped_data_path = OutputFileDatasetConfig(name="currency")

    aml_run_config = RunConfiguration()
    environment = work_space.environments["train_lstm"]
    aml_run_config.environment = environment

    get_currency = PythonScriptStep(
        name="get_currency",
        script_name="get_currency.py",
        compute_target="cpu-cluster",
        runconfig=aml_run_config,
        arguments=["--target_path", prepped_data_path],
        allow_reuse=True,
    )
    experiment = Experiment(work_space, "get_currency")

    pipeline = Pipeline(workspace=work_space, steps=[get_currency])
    run = experiment.submit(pipeline)
    run.wait_for_completion(show_output=True)
    run.publish_pipline(
        name="get_currency_pipeline",
        description="Get currency with pipeline",
        version="1.0",
    )


if __name__ == "__main__":
    main()