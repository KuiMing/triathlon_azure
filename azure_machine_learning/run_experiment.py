"""
Hello on Azure machine learning.
"""
import os
from azureml.core import Workspace, Experiment, ScriptRunConfig
from azureml.core.authentication import InteractiveLoginAuthentication

# import azureml


def main():
    """
    Hello on Azure machine learning.
    """
    interactive_auth = InteractiveLoginAuthentication(tenant_id=os.getenv("TENANT_ID"))
    work_space = Workspace.from_config(auth=interactive_auth)

    experiment = Experiment(workspace=work_space, name="hello-experiment")

    config = ScriptRunConfig(
        source_directory=".", script="hello.py", compute_target="cpu-cluster"
    )
    # azureml._restclient.snapshots_client.SNAPSHOT_MAX_SIZE_BYTES = 2000000000

    run = experiment.submit(config)
    aml_url = run.get_portal_url()
    print(aml_url)
    run.wait_for_completion(show_output=True)


if __name__ == "__main__":
    main()