"""
Run the experiment for training
"""
import os
from azureml.core import Workspace, Environment
from azureml.core.authentication import InteractiveLoginAuthentication


def main():
    """
    Run the experiment for training
    """
    interactive_auth = InteractiveLoginAuthentication(tenant_id=os.getenv("TENANT_ID"))
    work_space = Workspace.from_config(auth=interactive_auth)

    # Set up the Tensoflow/Keras environment
    environment = Environment.from_pip_requirements(
        name="train_lstm", file_path="requirements.txt"
    )
    environment.python.conda_dependencies.set_python_version("3.7.7")
    environment.register(work_space)


if __name__ == "__main__":
    main()
