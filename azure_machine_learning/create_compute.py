"""
Create a compute resource
"""
import os
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.authentication import InteractiveLoginAuthentication


def main():
    """
    Create a compute resource
    """
    # This automatically looks for a directory .azureml
    interactive_auth = InteractiveLoginAuthentication(tenant_id=os.getenv("TENANT_ID"))
    work_space = Workspace.from_config(auth=interactive_auth)

    # Choose a name for your CPU cluster
    cpu_cluster_name = "cpu-cluster"

    # Verify that the cluster does not exist already
    try:
        cpu_cluster = ComputeTarget(workspace=work_space, name=cpu_cluster_name)
        print("Found existing cluster, use it.")
    except ComputeTargetException:
        compute_config = AmlCompute.provisioning_configuration(
            vm_size="STANDARD_D2_V2", max_nodes=4, idle_seconds_before_scaledown=2400
        )
        cpu_cluster = ComputeTarget.create(work_space, cpu_cluster_name, compute_config)

    cpu_cluster.wait_for_completion(show_output=True)


if __name__ == "__main__":
    main()
