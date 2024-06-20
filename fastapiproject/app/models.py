from fastapi import  Form,  HTTPException
from pydantic import BaseModel
from pydantic import ValidationError

class PredictParam(BaseModel):
    """
    This class defines the parameters required for the query to the language model.
    """
    query: str



def parse_query_params(
        query: str = Form(..., description="The user's query sent to the LLM."),

):
    try:
        return PredictParam(
            query=query,
        )
    except ValidationError as err:
        raise HTTPException(status_code=400, detail=f"{err}") from err

