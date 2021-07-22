import os
from azureml.core import Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.pipeline.core import PublishedPipeline

from azureml.pipeline.core.schedule import ScheduleRecurrence, Schedule, TimeZone


def main():

    interactive_auth = InteractiveLoginAuthentication(tenant_id=os.getenv("TENANT_ID"))
    work_space = Workspace.from_config(auth=interactive_auth)
    pipelines = PublishedPipeline.list(work_space)
    pipeline_id = next(
        p_l.id for p_l in pipelines if p_l.name == "pipeline_data_train_deploy"
    )
    recurrence = ScheduleRecurrence(
        frequency="Week",
        interval=1,
        start_time="2021-07-21T07:00:00",
        time_zone=TimeZone.TaipeiStandardTime,
        week_days=["Sunday"],
        time_of_day="6:00",
    )
    Schedule.create(
        work_space,
        name="pipeline_data_train_deploy",
        description="Get data, train model and deploy service at 6:00 every Sunday",
        pipeline_id=pipeline_id,
        experiment_name="pipeline_data_train_deploy",
        recurrence=recurrence,
    )


if __name__ == "__main__":
    main()
