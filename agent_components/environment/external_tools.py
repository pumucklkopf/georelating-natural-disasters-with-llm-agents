import os
import pickle
import urllib.parse
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from agent_components.environment.internal_tools import OutputParser, ArticleSyntaxValidator
from agent_components.llms.chatAI import ChatAIHandler
from agent_components.memory.working_memory import WorkingMemory
from models.candidates import ToponymWithCandidates, CandidateGenerationOutput, ReflectionPhase
from models.errors import Error, ExecutionStep
from models.llm_output import ValidatedOutput


class GeoNamesAPI:
    def __init__(self, article_id: str = None):
        self.base_url = "http://api.geonames.org/search?"

    def search(self, params):
        params.update({'username': os.getenv('GEONAMES_USERNAME')})
        url = self.base_url + urllib.parse.urlencode(params)
        response = requests.get(url)
        return response.json()

    def retrieve_candidates(self, validated_output: ValidatedOutput) -> CandidateGenerationOutput:
        candidate_generation_output = CandidateGenerationOutput(**validated_output.model_dump())
        try:
            topos_to_search = validated_output.valid_toponyms
            correct_duplicates = validated_output.duplicate_toponyms
            if hasattr(validated_output, 'reflection_phase'):
                if validated_output.reflection_phase != ReflectionPhase.INITIAL_ACTOR_GENERATION:
                    topos_to_search = [topo for topo in validated_output.valid_toponyms if topo.generated_by_retry]
                    correct_duplicates = [topo for topo in validated_output.duplicate_toponyms if topo.generated_by_retry]
            for toponym_to_search_for in topos_to_search:
                response = self.search(toponym_to_search_for.params)
                toponym_with_candidates = ToponymWithCandidates(
                    toponym_with_search_arguments=toponym_to_search_for,
                    total_results = response['totalResultsCount'],
                    candidates=response['geonames'],
                    nof_retrieved_candidates=len(response['geonames'])
                )
                candidate_generation_output.toponyms_with_candidates.append(toponym_with_candidates)
            for duplicate_toponym in correct_duplicates:
                for toponym_with_candidates in candidate_generation_output.toponyms_with_candidates:
                    if toponym_with_candidates.toponym_with_search_arguments.toponym.casefold() == duplicate_toponym.duplicate_of.casefold():
                        candidate_generation_output.toponyms_with_candidates.append(
                            ToponymWithCandidates(
                                toponym_with_search_arguments=duplicate_toponym,
                                total_results=toponym_with_candidates.total_results,
                                candidates=toponym_with_candidates.candidates,
                                nof_retrieved_candidates=toponym_with_candidates.nof_retrieved_candidates
                            )
                        )
                        break
            return candidate_generation_output
        except Exception as e:
            # TODO: that's shit, because probably the generation was valid, but there is an issue with the external API
            # TODO: --> how to handle that?
            candidate_generation_output.fatal_errors = [Error(execution_step=ExecutionStep.GEOAPI,
                                                              error_message=str(e))]
            return candidate_generation_output

if __name__ == "__main__":
    article_ids = ['31767483', '44148889', '44228209'] #'31767483']
    load_dotenv()
    working_memory = WorkingMemory()
    prompt = working_memory.create_final_prompt()
    handler = ChatAIHandler()
    llm = handler.get_model("meta-llama-3.1-70b-instruct")
    geonames = GeoNamesAPI()
    for article_id in tqdm(article_ids, desc="Processing articles"):
        article = working_memory.few_shot_handler.data_handler.get_article_for_prompting(article_id)
        toponyms = working_memory.few_shot_handler.data_handler.get_short_toponyms_for_article(article_id)
        tops = str(toponyms)
        parser = OutputParser(article_id=article_id, toponym_list=toponyms)
        validator = ArticleSyntaxValidator()
        chain = prompt | llm | parser.extract_output | validator.validate_toponyms_of_article | geonames.retrieve_candidates
        llm_answer = chain.invoke(
            {
                "input__heading": article.get('title'),
                "input__news_article": article.get('text'),
                "input__toponym_list": tops
            }
        )
        with open(f'output/few_shot_candidate_generation_article_{article_id}.pkl', 'wb') as f:
            pickle.dump(llm_answer, f)
