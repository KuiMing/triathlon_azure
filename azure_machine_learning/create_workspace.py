"""
Create workspace
"""
import os
from azureml.core import Workspace

from azureml.core.authentication import InteractiveLoginAuthentication


def main():
    """
    Create workspace
    """
    interactive_auth = InteractiveLoginAuthentication(tenant_id=os.getenv("TENANT_ID"))

    work_space = Workspace.create(
        name="mltriathlon",  # provide a name for your workspace
        subscription_id=os.getenv("SUBSCRIPTION_ID"),  # provide your subscription ID
        resource_group="triathlon",  # provide a resource group name
        create_resource_group=True,
        location="eastus2",  # For example: 'westeurope', 'eastus2', 'westus2' or 'southeastasia'.
        auth=interactive_auth,
    )

    # write out the workspace details to a configuration file: .azureml/config.json
    work_space.write_config(path=".azureml")


if __name__ == "__main__":
    main()
