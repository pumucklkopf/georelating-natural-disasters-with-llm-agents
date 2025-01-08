import pickle
import os

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import END
from langgraph.graph import StateGraph
from tqdm import tqdm

from agent_components.environment.external_tools import GeoNamesAPI
from agent_components.environment.internal_tools import ArticleSyntaxValidator, OutputParser
from agent_components.llms.api_error_handler import handle_api_errors
from agent_components.llms.chatAI import ChatAIHandler
from agent_components.memory.working_memory import WorkingMemory
from models.candidates import CandidateGenerationState, CandidateGenerationOutput, ReflectionPhase
from models.errors import Error, ExecutionStep
from models.llm_output import LLMOutput, ValidatedOutput

string_seperator = "\n\n----------------------------------------\n"


class ReflectiveCandidateGenerator:
    def __init__(self, llm_model_name: str = "meta-llama-3.1-8b-instruct"):
        load_dotenv()
        self.working_memory = WorkingMemory()
        self.data_handler = self.working_memory.few_shot_handler.data_handler
        self.llm_handler = ChatAIHandler()
        self.llm = self.llm_handler.get_model(llm_model_name)
        self.validator = ArticleSyntaxValidator()
        self.geonames = GeoNamesAPI()
        self.call_times = []

    """
    Node functions
    """

    def create_prompt(self, input_state: CandidateGenerationState) -> CandidateGenerationState:
        toponyms = self.data_handler.get_short_toponyms_for_article(input_state.article_id)
        prompt = self.working_memory.create_final_prompt()
        return CandidateGenerationState(
            article_id=input_state.article_id,
            article_title=input_state.article_title,
            article_text=input_state.article_text,
            toponyms=toponyms,
            reflection_phase=ReflectionPhase.INITIAL_ACTOR_GENERATION,
            initial_prompt=prompt
        )

    def call_actor(self, state: CandidateGenerationState) -> LLMOutput:
        # First check in which state of the reflective candidate generation we are
        if state.reflection_phase == ReflectionPhase.INITIAL_ACTOR_GENERATION:
            prompt = state.initial_prompt
        elif state.reflection_phase == ReflectionPhase.REFLECTED_ACTOR_GENERATION:
            prompt = state.reflected_prompt
        else:
            raise ValueError(f"Invalid reflection phase: {state.reflection_phase}")
        # Build the LLM call sub-chain
        chain = prompt | self.llm

        @handle_api_errors(call_times=self.call_times)
        def _invoke_chain(input_dict: dict):
            return chain.invoke(input_dict)

        try:
            llm_answer = _invoke_chain({
                "input__heading": state.article_title,
                "input__news_article": state.article_text,
                "input__toponym_list": str(state.toponyms)
            })

            return LLMOutput(
                article_id=state.article_id,
                toponyms=state.toponyms,
                raw_output=AIMessage(**llm_answer.model_dump())
            )
        except Exception as e:
            return LLMOutput(
                article_id=state.article_id,
                toponyms=state.toponyms,
                fatal_errors=[Error(
                    execution_step=ExecutionStep.ACTOR,
                    error_message=str(e)
                )]
            )

    def extract_output(self, state: LLMOutput) -> LLMOutput:
        parser = OutputParser(article_id=state.article_id,
                              toponym_list=state.toponyms)
        parsed_output = parser.extract_output(state.raw_output)
        # Ensure no duplicate keys by unpacking parsed_output selectively
        parsed_data = parsed_output.model_dump()
        parsed_data.update({
            "article_title": state.article_title,
            "article_text": state.article_text,
        })
        return LLMOutput(**parsed_data)

    def validate_output(self, state: LLMOutput) -> ValidatedOutput:
        return self.validator.validate_toponyms_of_article(state)

    def retrieve_candidates(self, state: ValidatedOutput) -> CandidateGenerationOutput:
        return self.geonames.retrieve_candidates(state)

    def criticize(self, state: CandidateGenerationState) -> CandidateGenerationState:
        if state.reflection_phase == ReflectionPhase.REFLECTED_ACTOR_GENERATION:
            state.fatal_errors.append(Error(
                execution_step=ExecutionStep.CRITIC,
                error_message="Critique did not solve fatal errors. Exiting."
            ))
            return state
        else:
            state.reflection_phase = ReflectionPhase.CRITIC_GENERATION

        def _generate_initial_generation_prompt_text() -> str:
            original_prompt_text = state.initial_prompt.format(
                input__heading=state.article_title,
                input__news_article=state.article_text,
                input__toponym_list=str(state.toponyms)
            )
            generated_output = str(state.raw_output.content)
            fatal_errors = str([error.model_dump() for error in state.fatal_errors])

            initial_generation_prompt_text = (
                f"Original Prompt Text: \n {original_prompt_text + string_seperator}"
                f"Generated Output: \n {generated_output+ string_seperator}\n\n"
                f"Fatal Errors: \n {fatal_errors}")
            return initial_generation_prompt_text

        def _generate_critic_prompt() -> PromptTemplate:
            critic_system_prompt = (
                "System: \n"
                "You are a constructive critic for the actor LLM which generated the output below. Please analyze the "
                "occurring errors in the generated output and provide actionable feedback to fix them. Your feedback "
                "will be directly used to guide the actor LLM to generate better outputs in the future. Only provide "
                "feedback to the errors, and be as concise as possible."
                # Actionable feedback, c.g. Madaan et al. (2023) and Lauscher et al. (2024)
            )
            initial_generation_prompt = _generate_initial_generation_prompt_text()
            critic_instruction = "Your feedback:\n"

            critic_prompt_text = (critic_system_prompt + string_seperator +
                                  initial_generation_prompt + string_seperator +
                                  critic_instruction)
            return PromptTemplate(
                template=critic_prompt_text,
                template_format="mustache",
                input_variables=[]
            )

        def _generate_reflected_actor_prompt() -> PromptTemplate:
            reflective_actor_system_prompt = (
                "System: \n"
                "You are a reflective actor. Please strictly follow the feedback provided by the critic to generate a "
                "new output which does not lead to the errors your previous generation caused."
            )
            initial_generation_prompt_text = _generate_initial_generation_prompt_text()
            feedback = f"Actionable feedback by the critic: \n{str(state.critic_feedback.content)}"
            reflective_actor_instruction_prompt_text = "Your new output in plain JSON:\n"

            reflected_prompt_text = (reflective_actor_system_prompt + string_seperator +
                                     initial_generation_prompt_text + string_seperator +
                                     feedback + string_seperator +
                                     reflective_actor_instruction_prompt_text)
            return PromptTemplate(
                template=reflected_prompt_text,
                template_format="mustache",
                input_variables=[]
            )

        @handle_api_errors(call_times=self.call_times)
        def _invoke_critic(lang_chain):
            return lang_chain.invoke({})

        # compile final critic prompt and chain
        state.critic_prompt = _generate_critic_prompt()
        chain = state.critic_prompt | self.llm

        try:
            # invoke critic and use feedback to generate reflected actor prompt
            critic_feedback = _invoke_critic(chain)
            state.critic_feedback = AIMessage(**critic_feedback.model_dump())
            state.reflected_prompt = _generate_reflected_actor_prompt()
            state.reflection_phase = ReflectionPhase.REFLECTED_ACTOR_GENERATION

            # Reset the state for the next iteration of the reflective candidate generation
            state.parsed_output = state.valid_toponyms = state.duplicate_toponyms = state.toponyms_with_candidates = state.fatal_errors = []
            return state

        except Exception as e:
            state.fatal_errors.append(Error(
                execution_step=ExecutionStep.CRITIC,
                error_message=str(e)
            ))
            return state

    """
    Routing functions
    """

    def has_fatal_errors(self, state: CandidateGenerationState) -> str:
        if state.fatal_errors:
            return "has_fatal_errors"
        else:
            return "successful"

    def critique_was_successful(self, state: CandidateGenerationState) -> str:
        if state.fatal_errors:
            if state.fatal_errors[-1].execution_step == ExecutionStep.CRITIC:
                return "critique_unsuccessful"
        else:
            return "critique_successful"

    """
    Grap Structure
    """

    def build_graph(self):
        graph_builder = StateGraph(CandidateGenerationState)

        # Add nodes to the graph
        graph_builder.add_node("create_prompt", self.create_prompt)
        graph_builder.add_node("call_actor", self.call_actor)
        graph_builder.add_node("extract_output", self.extract_output)
        graph_builder.add_node("validate_output", self.validate_output)
        graph_builder.add_node("retrieve_candidates", self.retrieve_candidates)
        graph_builder.add_node("criticize", self.criticize)

        # Set the entry point and define edges
        graph_builder.set_entry_point("create_prompt")
        graph_builder.add_edge("create_prompt", "call_actor")
        graph_builder.add_conditional_edges("call_actor",
                                            self.has_fatal_errors,
                                            {
                                                "has_fatal_errors": "criticize",
                                                "successful": "extract_output"
                                            })
        graph_builder.add_conditional_edges("extract_output",
                                            self.has_fatal_errors,
                                            {
                                                "has_fatal_errors": "criticize",
                                                "successful": "validate_output"
                                            })
        graph_builder.add_conditional_edges("validate_output",
                                            self.has_fatal_errors,
                                            {
                                                "has_fatal_errors": "criticize",
                                                "successful": "retrieve_candidates"
                                            })
        graph_builder.add_conditional_edges("retrieve_candidates",
                                            self.has_fatal_errors,
                                            {
                                                "has_fatal_errors": "criticize",
                                                "successful": END
                                            })
        graph_builder.add_conditional_edges("criticize",
                                            self.critique_was_successful,
                                            {
                                                "critique_successful": "call_actor",
                                                "critique_unsuccessful": END
                                            })

        return graph_builder

    """
    Compile and run the graph
    """

    def run_graph(self, article: pd.Series):
        graph_builder = self.build_graph()
        agent_graph = graph_builder.compile()

        # Run the graph
        input_state = {
            "article_id": article['docid'],
            "article_title": article['title'],
            "article_text": article['text']
        }
        agent_graph_answer = agent_graph.invoke(input_state)

        return agent_graph_answer

    def reflectively_generate_candidates_for_evaluation(self,
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

            agent_graph_answer = self.run_graph(article)

            # Parse dictionary (default langgraph output) to CandidateGenerationState
            agent_graph_answer = CandidateGenerationState(**agent_graph_answer)

            # Necessary to format to be able to serialize the object
            agent_graph_answer.initial_prompt = agent_graph_answer.initial_prompt.format(
                input__heading=agent_graph_answer.article_title,
                input__news_article=agent_graph_answer.article_text,
                input__toponym_list=str(agent_graph_answer.toponyms)
            )

            with open(os.path.join(output_dir, f'{article_id}.pkl'), 'wb') as f:
                pickle.dump(agent_graph_answer, f)

    def generate_graph_image(self, output_file_path: str = "graph_layout_image.png"):
        graph_builder = self.build_graph()
        agent_graph = graph_builder.compile()
        image_data = agent_graph.get_graph().draw_mermaid_png(output_file_path=output_file_path)
        # Open the image
        os.system("graph_layout_image.png")
        return image_data


if __name__ == "__main__":
    generator = ReflectiveCandidateGenerator(
        llm_model_name="meta-llama-3.1-8b-instruct"
    )
    generator.generate_graph_image()
    # generator.reflectively_generate_candidates_for_evaluation(
    #     seed=42,
    #     nof_articles=100,
    #     output_dir=f'output/reflective_candidate_generation/{pd.Timestamp.now().strftime("%Y%m%d")}_seed_42_100_articles'
    # )
