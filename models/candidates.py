from enum import Enum

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from typing import List, Dict

from models.errors import Error
from models.llm_output import ToponymSearchArguments, ValidatedOutput


# enum of the possible phases of the reflective candidate generation
class ReflectionPhase(str, Enum):
    INITIAL_ACTOR_GENERATION = "initial_actor_generation"
    CRITIC_GENERATION_FOR_FATAL_ERRORS = "critic_generation_for_fatal_errors"
    CRITIC_GENERATION_FOR_INVALID_TOPONYMS = "critic_generation_for_invalid_toponyms"
    ACTOR_RETRY_AFTER_FATAL_ERROR = "actor_retry_after_fatal_error"
    ACTOR_RETRY_ON_INVALID_TOPONYMS = "actor_retry_on_invalid_toponyms"
    FATAL_ERRORS_NOT_SOLVED = "fatal_errors_not_solved"
    RESOLUTION_ACTOR_GENERATION = "resolution_actor_generation"
    RESOLUTION_ACTOR_RETRY_ON_INVALID_RESOLUTIONS = "resolution_actor_retry_on_invalid_resolutions"
    RESOLUTION_ACTOR_RETRY_AFTER_FATAL_ERROR = "resolution_actor_retry_after_fatal_error"
    RESOLUTION_CRITIC_GENERATION_FOR_INVALID_RESOLUTIONS = "resolution_critic_generation_for_invalid_resolutions"
    RESOLUTION_CRITIC_GENERATION_FOR_FATAL_ERRORS = "resolution_critic_generation_for_fatal_errors"


class ToponymWithCandidates(BaseModel):
    toponym_with_search_arguments: ToponymSearchArguments = Field(description="The toponym and search arguments")
    total_results: int = Field(
        description="Total number of results  from the GeoNames API matching the search arguments",
        default=0)
    candidates: List[Dict] = Field(description="List of candidates retrieved from the GeoNames API",
                                   default=[])
    nof_retrieved_candidates: int = Field(description="Number of candidates retrieved from the GeoNames API",
                                          default=0)


class CandidateGenerationOutput(ValidatedOutput):
    toponyms_with_candidates: List[ToponymWithCandidates] = Field(description="List of toponyms with candidates",
                                                                  default=[])


class ToponymWithCandidatesShort(BaseModel):
    toponym: str = Field(description="The toponym for which the candidates were retrieved")
    candidates: List[Dict] = Field(description="List of candidates retrieved from the GeoNames API")


class CandidateResolutionInput(BaseModel):
    toponyms_with_candidates: List[ToponymWithCandidatesShort] = Field(
        description="List of toponyms with candidates",
        default=[]
    )


class CandidateGenerationState(CandidateGenerationOutput):
    article_title: str = Field(description="The title of the article",
                               default="")
    article_text: str = Field(description="The text of the article",
                              default="")
    initial_prompt: str = Field(description="The prompt used to generate the output",
                                default="")
    reflection_phase: ReflectionPhase = Field(description="The phase of the reflective candidate generation",
                                              default=ReflectionPhase.INITIAL_ACTOR_GENERATION)
    critic_prompt: str = Field(description="The prompt used to generate the critic output",
                               default="")
    critic_feedback: AIMessage = Field(description="The output (feedback) generated by the critic",
                                       default=AIMessage(content=""))
    reflected_prompt: str = Field(
        description="The prompt containing actionable feedback by the critic",
        default="")


class ResolvedToponym(BaseModel):
    toponym: str = Field(description="The toponym to resolve")
    reasoning: str = Field(description="The reasoning for the selection of a candidate")
    selected_candidate_geonameId: int | None = Field(description="The GeoNames ID of the selected candidate")


class ResolvedToponymWithErrors(ResolvedToponym):
    errors: List[Error] = Field(description="List of errors in the resolution of the toponym")


class GeoCodingState(CandidateGenerationState):
    resolution_initial_prompt: str = Field(description="The prompt used to generate the resolution output",
                                           default="")
    resolution_basic_instructions: str = Field(description="The basic instructions for the resolution model",
                                               default="")
    resolution_raw_output: AIMessage = Field(description="The raw output of the resolution model",
                                             default=AIMessage(content=""))
    geocoded_toponyms: List[ResolvedToponym] = Field(description="List of resolved toponyms",
                                                     default=[])
    valid_geocoded_toponyms: List[ResolvedToponym] = Field(description="List of valid resolved toponyms",
                                                           default=[])
    invalid_geocoded_toponyms: List[ResolvedToponymWithErrors] = Field(description="List of invalid resolved toponyms",
                                                                       default=[])
    resolution_fatal_errors: List[Error] = Field(description="List of fatal errors in the resolution output",
                                                 default=[])
    resolution_critic_prompt: str = Field(description="The prompt used to generate the resolution critic output",
                                          default="")
    resolution_critic_feedback: AIMessage = Field(
        description="The output (feedback) generated by the resolution critic",
        default=AIMessage(content=""))
    resolution_reflected_prompt: str = Field(
        description="The prompt containing actionable feedback by the resolution critic",
        default="")
