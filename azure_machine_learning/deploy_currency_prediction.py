"""
Deploy model to your service
"""
import numpy as np
from azureml.core import Model, Workspace
from azureml.core import Run
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice


def main():
    """
    Deploy model to your service
    """
    run = Run.get_context()
    try:
        work_space = run.experiment.workspace
    except AttributeError:
        work_space = Workspace.from_config()
    environment = work_space.environments["train_lstm"]
    model = Model(work_space, "currency")
    model_list = model.list(work_space)
    validation_accuracy = []
    version = []
    for i in model_list:
        validation_accuracy.append(float(i.properties["val_accuracy"]))
        version.append(i.version)
    model = Model(
        work_space, "currency", version=version[np.argmax(validation_accuracy)]
    )
    service_name = "currency-service"
    inference_config = InferenceConfig(
        entry_script="predict_currency.py", environment=environment
    )
    aci_config = AciWebservice.deploy_configuration(cpu_cores=1, memory_gb=1)
    service = Model.deploy(
        workspace=work_space,
        name=service_name,
        models=[model],
        inference_config=inference_config,
        deployment_config=aci_config,
        overwrite=True,
    )
    service.wait_for_deployment(show_output=True)
    print(service.get_logs())


if __name__ == "__main__":
    main()
