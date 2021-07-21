import os
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core import Workspace, Experiment, Dataset
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.pipeline.core import Pipeline


def main():
    interactive_auth = InteractiveLoginAuthentication(tenant_id=os.getenv("TENANT_ID"))
    work_space = Workspace.from_config(auth=interactive_auth)
    datastore = work_space.get_default_datastore()
    input_folder = (
        Dataset.File.from_files(path=(datastore, "currency"))
        .as_named_input("input_folder")
        .as_mount()
    )
    dataset = OutputFileDatasetConfig(
        name="usd_twd", destination=(datastore, "currency")
    )
    aml_run_config = RunConfiguration()
    environment = work_space.environments["train_lstm"]
    aml_run_config.environment = environment

    get_currency = PythonScriptStep(
        source_directory=".",
        name="get_currency",
        script_name="get_currency.py",
        compute_target="cpu-cluster",
        runconfig=aml_run_config,
        inputs=[input_folder],
        arguments=[
            "--target_folder",
            dataset.as_upload(overwrite=True).register_on_complete(name="currency"),
            "--input",
            input_folder,
        ],
        allow_reuse=True,
    )

    training = PythonScriptStep(
        source_directory=".",
        name="train_lstm",
        script_name="train_lstm.py",
        compute_target="cpu-cluster",
        runconfig=aml_run_config,
        inputs=[dataset.as_input()],
        arguments=["--target_folder", dataset.as_input()],
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
    experiment = Experiment(work_space, "pipeline_data_train_deploy")

    pipeline = Pipeline(workspace=work_space, steps=[get_currency, training, deploy])
    run = experiment.submit(pipeline)
    run.wait_for_completion(show_output=True)
    run.publish_pipeline(
        name="pipeline_data_train_deploy",
        description="data-->train-->deploy",
        version="1.0",
    )


if __name__ == "__main__":
    main()
