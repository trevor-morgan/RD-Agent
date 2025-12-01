
from rdagent.components.coder.CoSTEER.task import CoSTEERTask


# Because we use isinstance to distinguish between different types of tasks, we need to use sub classes to represent different types of tasks
class WorkflowTask(CoSTEERTask):
    def __init__(self, name: str = "Workflow", *args, **kwargs) -> None:
        super().__init__(name=name, *args, **kwargs)
