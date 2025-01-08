import os
import pickle
import time

import pandas as pd
from dotenv import load_dotenv
from openai import APITimeoutError, InternalServerError
from tqdm import tqdm

from agent_components.environment.external_tools import GeoNamesAPI
from agent_components.environment.internal_tools import OutputParser, ArticleSyntaxValidator
from agent_components.llms.api_error_handler import handle_api_errors
from agent_components.llms.chatAI import ChatAIHandler
from agent_components.memory.working_memory import WorkingMemory
from models.candidates import CandidateGenerationOutput
from models.errors import Error, ExecutionStep


class CandidateGenerator:
    def __init__(self, llm_model_name: str = "meta-llama-3.1-8b-instruct"):
        load_dotenv()
        self.working_memory = WorkingMemory()
        self.data_handler = self.working_memory.few_shot_handler.data_handler
        self.llm_handler = ChatAIHandler()
        self.llm = self.llm_handler.get_model(llm_model_name)
        self.validator = ArticleSyntaxValidator()
        self.geonames = GeoNamesAPI()
        self.call_times = []

    def generate_candidates_for_article(self, article: pd.Series):
        article_id = article['docid']
        toponyms = self.data_handler.get_short_toponyms_for_article(article_id)
        parser = OutputParser(article_id=article_id, toponym_list=toponyms)

        # Build the processing chain
        chain = (
            self.working_memory.create_final_prompt()
            | self.llm
            | parser.extract_output
            | self.validator.validate_toponyms_of_article
            | self.geonames.retrieve_candidates
        )

        @handle_api_errors(call_times=self.call_times)
        def _invoke_chain(input_dict: dict):
            return chain.invoke(input_dict)


        llm_answer = _invoke_chain({
            "input__heading": article['title'],
            "input__news_article": article['text'],
            "input__toponym_list": str(toponyms)
        })

        return llm_answer

    def generate_candidates_for_evaluation(self,
                                           seed: int = 42,
                                           nof_articles: int = 100,
                                           output_dir: str = 'output/'):
        os.makedirs(output_dir, exist_ok=True)

        articles_df = self.data_handler.get_random_articles_for_evaluation(
            seed=seed, n=nof_articles
        )
        for index, article in tqdm(articles_df.iterrows(), total=len(articles_df), desc="Processing articles"):
            # First check if the article has already been processed
            article_id = article['docid']

            if os.path.exists(os.path.join(output_dir, f'{article_id}.pkl')):
                continue

            content_to_save = None

            llm_answer = self.generate_candidates_for_article(article)

            if isinstance(llm_answer, APITimeoutError) or isinstance(llm_answer, InternalServerError):
                content_to_save = CandidateGenerationOutput(
                    article_id=article_id,
                    fatal_errors=[Error(
                        execution_step=ExecutionStep.LLM,
                        error_message=str(llm_answer)
                    )]

                )
            elif isinstance(llm_answer, CandidateGenerationOutput):
                content_to_save = llm_answer

            if content_to_save:
                with open(os.path.join(output_dir, f'{article_id}.pkl'), 'wb') as f:
                    pickle.dump(content_to_save, f)


if __name__ == "__main__":
    seed = 42
    nof_articles = 100
    candidate_generator = CandidateGenerator(
       llm_model_name="mistral-large-instruct"
    )
    start = time.time()
    candidate_generator.generate_candidates_for_evaluation(
        seed=seed,
        nof_articles=nof_articles,
        output_dir=f'output/baseline_candidate_generation/mistral-large/{pd.Timestamp.now().strftime("%Y%m%d")}_seed_{seed}_{nof_articles}_articles')
    print(f"Processing {nof_articles} articles took {time.time() - start:.2f} seconds.")