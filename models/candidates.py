from pydantic import BaseModel, Field
from typing import List, Dict

from models.llm_output import ToponymSearchArguments, ValidatedOutput


class ToponymWithCandidates(BaseModel):
    toponym_with_search_arguments: ToponymSearchArguments = Field(description="The toponym and search arguments")
    total_results: int = Field(description="Total number of results  from the GeoNames API matching the search arguments",
                               default=0)
    candidates: List[Dict] = Field(description="List of candidates retrieved from the GeoNames API",
                                   default=[])
    nof_retrieved_candidates: int = Field(description="Number of candidates retrieved from the GeoNames API",
                                          default=0)


class CandidateGenerationOutput(ValidatedOutput):
    toponyms_with_candidates: List[ToponymWithCandidates] = Field(description="List of toponyms with candidates",
                                                                  default=[])
