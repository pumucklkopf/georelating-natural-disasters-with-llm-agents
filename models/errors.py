from enum import Enum

from pydantic import BaseModel, Field


class ExecutionStep(Enum):
    SEARCHOUTPUTPARSER = "search_output_parser"
    SEARCHPARAMETERVALIDATOR = "search_parameter_validator"
    ARTICLESYNTAXVALIDATOR = "article_syntax_validator"
    GEOAPI = "geonames_search_api"
    LLM = "llm_generation"

class Error(BaseModel):
    execution_step: ExecutionStep = Field(description="The step in which the error occurred")
    error_message: str = Field(description="The error message")