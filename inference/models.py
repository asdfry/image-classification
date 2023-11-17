from pydantic import BaseModel
from typing_extensions import Annotated


class Result(BaseModel):
    label: Annotated[str, "Label that predicted"]
    elapsed_time: Annotated[float, "Elapsed time"]
    used_memory: Annotated[float, "Memory used"]
