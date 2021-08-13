"""
Hello on Azure machine learning.
"""
import os
from azureml.core import Workspace, Experiment, ScriptRunConfig, Environment
from azureml.core.authentication import InteractiveLoginAuthentication


def main():
    """
    Hello on Azure machine learning.
    """
    interactive_auth = InteractiveLoginAuthentication(tenant_id=os.getenv("TENANT_ID"))
    work_space = Workspace.from_config(auth=interactive_auth)

    experiment = Experiment(workspace=work_space, name="hello-experiment")
    environment = Environment("Experiment hello-experiment Environment")
    environment.docker.enabled = True
    environment.docker.base_image = "mcr.microsoft.com/azureml/intelmpi2018.3-ubuntu16.04:20210301.v1"
    environment.python.user_managed_dependencies = True
    config = ScriptRunConfig(
        source_directory=".", script="hello.py", compute_target="cpu-cluster",
        environment=environment
    )
    # azureml._restclient.snapshots_client.SNAPSHOT_MAX_SIZE_BYTES = 2000000000

    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print(aml_url)
    run.wait_for_completion(show_output=True)


if __name__ == "__main__":
    main()
