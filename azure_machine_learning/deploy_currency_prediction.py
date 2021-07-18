"""
Deploy model to your service
"""
import os
import numpy as np
from azureml.core import Model, Workspace
from azureml.core import Run
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice
from azureml.core.authentication import InteractiveLoginAuthentication


def main():
    """
    Deploy model to your service
    """
    interactive_auth = InteractiveLoginAuthentication(tenant_id=os.getenv("TENANT_ID"))

    run = Run.get_context()
    try:
        work_space = run.experiment.workspace
    except AttributeError:
        work_space = Workspace.from_config(auth=interactive_auth)
    environment = work_space.environments["train_lstm"]
    model = Model(work_space, "currency")
    model_list = model.list(work_space, name="currency")
    val_loss = []
    version = []
    for i in model_list:
        # print(i)
        val_loss.append(float(i.properties["val_loss"]))
        version.append(i.version)
    model = Model(work_space, "currency", version=version[np.argmin(val_loss)])
    service_name = "currency-service"
    inference_config = InferenceConfig(
        entry_script="predict_currency.py", environment=environment
    )
    aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
    scaler = Model(work_space, name="scaler", version=1)
    service = Model.deploy(
        workspace=work_space,
        name=service_name,
        models=[model, scaler],
        inference_config=inference_config,
        deployment_config=aci_config,
        overwrite=True,
    )
    service.wait_for_deployment(show_output=True)
    print(service.get_logs())
    print(service.scoring_uri)


if __name__ == "__main__":
    main()
