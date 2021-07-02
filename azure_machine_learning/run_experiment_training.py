"""
Run the experiment for training
"""
import os
import argparse
from azureml.core import ScriptRunConfig, Dataset, Workspace, Experiment, Environment
from azureml.core.model import Model
from azureml.tensorboard import Tensorboard
from azureml.core.authentication import InteractiveLoginAuthentication


def parse_args():
    """
    Parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="python script", type=str)
    parser.add_argument(
        "-t", "--target_path", help="file folder in datastore", type=str
    )
    parser.add_argument("-n", "--experiment_name", help="name of experiment", type=str)
    args = parser.parse_args()
    return args


def main():
    """
    Run the experiment for training
    """
    args = parse_args()
    interactive_auth = InteractiveLoginAuthentication(tenant_id=os.getenv("TENANT_ID"))
    work_space = Workspace.from_config(auth=interactive_auth)

    # Set up the dataset for training
    datastore = work_space.get_default_datastore()
    dataset = Dataset.File.from_files(path=(datastore, args.target_path))

    # Set up the experiment for training
    experiment = Experiment(workspace=work_space, name=args.experiment_name)
    #     azureml._restclient.snapshots_client.SNAPSHOT_MAX_SIZE_BYTES = 2000000000
    config = ScriptRunConfig(
        source_directory=".",
        script=args.file,
        compute_target="cpu-cluster",
        arguments=[
            "--data_folder",
            dataset.as_named_input("input").as_mount(),
            "--log_folder",
            "./logs",
        ],
    )

    # Set up the Tensoflow/Keras environment
    environment = Environment.from_pip_requirements(
        name=args.file.replace(".py", ""), file_path="requirements.txt"
    )
    environment.register(work_space)
    config.run_config.environment = environment

    # Run the experiment for training
    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print(
        "Submitted to an Azure Machine Learning compute cluster. Click on the link below"
    )
    print("")
    print(aml_url)

    tboard = Tensorboard([run])
    # If successful, start() returns a string with the URI of the instance.
    tboard.start(start_browser=True)
    run.wait_for_completion(show_output=True)
    # After your job completes, be sure to stop() the streaming otherwise it will continue to run.
    print("Press enter to stop")
    input()
    tboard.stop()

    # Register Model
    metrics = run.get_metrics()
    run.register_model(
        model_name=args.experiment_name,
        tags={"data": "mnist", "model": "classification"},
        model_path="outputs/keras_lenet.h5",
        model_framework=Model.Framework.TENSORFLOW,
        model_framework_version="2.3.1",
        properties={
            "train_loss": metrics["train_loss"][-1],
            "train_accuracy": metrics["train_accuracy"][-1],
            "val_loss": metrics["val_loss"][-1],
            "val_accuracy": metrics["val_accuracy"][-1],
        },
    )


if __name__ == "__main__":
    main()
