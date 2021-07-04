import os
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core import Workspace, Experiment, Dataset
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.pipeline.core import Pipeline


def main():
    interactive_auth = InteractiveLoginAuthentication(tenant_id=os.getenv("TENANT_ID"))
    work_space = Workspace.from_config(auth=interactive_auth)
    datastore = work_space.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, "currency"))

    aml_run_config = RunConfiguration()
    environment = work_space.environments["train_lstm"]
    aml_run_config.environment = environment

    training = PythonScriptStep(
        source_directory=".",
        name="train_lstm",
        script_name="train_lstm.py",
        compute_target="cpu-cluster",
        runconfig=aml_run_config,
        arguments=["--target_path", dataset.as_named_input("input").as_mount()],
        allow_reuse=True,
    )
    deploy = PythonScriptStep(
        source_directory=".",
        name="deploy_currency_prediction",
        script_name="deploy_currency_prediction.py",
        compute_target="cpu-cluster",
        runconfig=aml_run_config,
        allow_reuse=True,
    )

    experiment = Experiment(work_space, "train_deploy")

    pipeline = Pipeline(workspace=work_space, steps=[training, deploy])
    run = experiment.submit(pipeline)
    run.wait_for_completion(show_output=True)
    run.publish_pipline(
        name="train_deploy_pipeline",
        description="train the lstm model and deploy the prediction service with pipeline",
        version="1.0",
    )


if __name__ == "__main__":
    main()
