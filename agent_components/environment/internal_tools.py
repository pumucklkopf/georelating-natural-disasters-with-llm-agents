import json
import re
import itertools

import pycountry
from dotenv import load_dotenv
from langchain_core.messages import AIMessage

from agent_components.llms.chatAI import ChatAIHandler
from agent_components.memory.working_memory import WorkingMemory
from models.candidates import ReflectionPhase
from models.errors import Error, ExecutionStep
from models.llm_output import ToponymSearchArguments, ToponymSearchArgumentsWithErrors, ValidatedOutput, LLMOutput


def _is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def _is_integer(value):
    try:
        int(value)
        return True
    except ValueError:
        return False


def _is_valid_countrycode(code):
    return pycountry.countries.get(alpha_2=code.upper()) is not None


class OutputParser:
    def __init__(self, article_id: str, toponym_list: list[str]):
        self.article_id = article_id
        self.toponym_list = toponym_list

    def extract_output(self, message: AIMessage) -> LLMOutput:
        parsed_output = []
        llm_output = LLMOutput(
            article_id=self.article_id,
            toponyms=self.toponym_list,
            raw_output=message.model_copy(),
            parsed_output=[]
        )
        try:
            # if the string does not start with a '[', strip of the characters before the first '['
            if message.content[0] != '[':
                message.content = message.content[message.content.find('['):]
                if message.content[-1] != ']':
                    message.content = message.content[:message.content.rfind(']') + 1]
            for location_mention in json.loads(message.content):
                parsed_output.append(ToponymSearchArguments.model_validate(location_mention))
            llm_output.parsed_output.extend(parsed_output)
            return llm_output
        except Exception as e:
            error = f"Error parsing output or loading JSON format: {e}"
            llm_output.fatal_errors = [Error(execution_step=ExecutionStep.SEARCHOUTPUTPARSER, error_message=error)]
            return llm_output


class SearchParameterSyntaxValidator:
    def __init__(self):
        self.params = {}
        self.errors = []
        self.required_params = ['q', 'name', 'name_equals']
        self.optional_params = [
            'name_startsWith', 'maxRows', 'startRow', 'country', 'countryBias',
            'continentCode', 'adminCode1', 'adminCode2', 'adminCode3', 'adminCode4',
            'adminCode5', 'featureClass', 'featureCode', 'cities', 'lang', 'type',
            'style', 'isNameRequired', 'tag', 'operator', 'charset', 'fuzzy',
            'east', 'west', 'north', 'south', 'searchlang', 'orderby', 'inclBbox'
        ]
        self.continent_codes = ['AF', 'AS', 'EU', 'NA', 'OC', 'SA', 'AN']

    def validate(self, params) -> tuple[bool, list[str]]:
        """
        Validate the syntax of the GeoNames API parameters
        :param params: dict
        :return: tuple[bool, list[str]] - (is_valid, errors)
        """
        self.params = params
        self.errors = []
        self.validate_required_params()
        self.validate_optional_params()
        self.validate_bounding_box()
        self.validate_fuzzy()
        self.validate_maxRows()
        self.validate_startRow()
        self.validate_country()
        self.validate_featureClass()
        self.validate_featureCode()
        self.validate_cities()
        self.validate_lang()
        self.validate_type()
        self.validate_style()
        self.validate_isNameRequired()
        self.validate_tag()
        self.validate_operator()
        self.validate_charset()
        self.validate_orderby()
        self.validate_inclBbox()

        return len(self.errors) == 0, self.errors

    def validate_required_params(self):
        if not any(param in self.params for param in self.required_params):
            self.errors.append("One of 'q', 'name', or 'name_equals' is required.")
        else:
            for param in self.required_params:
                if param in self.params:
                    if not isinstance(self.params[param], str):
                        self.errors.append(f"{param} must be a string.")

    def validate_optional_params(self):
        for param in self.params:
            if param not in self.required_params + self.optional_params:
                self.errors.append(f"Invalid parameter: {param}")

    def validate_name_startsWith(self):
        if 'name_startsWith' in self.params and not isinstance(self.params['name_startsWith'], str):
            self.errors.append("name_startsWith must be a string.")

    def validate_maxRows(self):
        if 'maxRows' in self.params and not _is_integer(self.params['maxRows']):
            self.errors.append("maxRows must be an integer.")

    def validate_startRow(self):
        if 'startRow' in self.params:
            if not _is_integer(self.params['startRow']):
                self.errors.append("startRow must be an integer.")
            elif not 0 <= int(self.params['startRow']) <= 5000:
                self.errors.append("startRow must be non-negative and maximum 5000.")

    def validate_country(self):
        if 'country' in self.params and not _is_valid_countrycode(self.params['country']):
            self.errors.append("country must be a ISO 3166-1 Alpha-2 Code.")

    def validate_countryBias(self):
        if 'countryBias' in self.params and not _is_valid_countrycode(self.params['countryBias']):
            self.errors.append("countryBias must be a ISO 3166-1 Alpha-2 Code.")

    def validate_continentCode(self):
        if 'continentCode' in self.params and self.params not in self.continent_codes:
            self.errors.append(f"continentCode must be one of {self.continent_codes}.")

    def validate_featureClass(self):
        if 'featureClass' in self.params and not re.match(r'^[AHLPRSTUV]+$', self.params['featureClass']):
            self.errors.append("featureClass must be one or more of A,H,L,P,R,S,T,U,V.")

    def validate_featureCode(self):
        # ToDo: Check if featureCode is a valid code according to https://www.geonames.org/export/codes.html
        if 'featureCode' in self.params and not isinstance(self.params['featureCode'], str):
            self.errors.append("featureCode must be a string.")

    def validate_cities(self):
        if 'cities' in self.params and self.params['cities'] not in ['cities1000', 'cities5000', 'cities15000']:
            self.errors.append("cities must be one of 'cities1000', 'cities5000', 'cities15000'.")

    def validate_lang(self):
        if 'lang' in self.params:
            self.errors.append("language shall always be English.")

    def validate_type(self):
        if 'type' in self.params and self.params['type'] != 'json':
            self.errors.append("type shall always be json.")

    def validate_style(self):
        if 'style' in self.params and self.params['style'] not in ['SHORT', 'MEDIUM', 'LONG', 'FULL']:
            self.errors.append("style must be one of 'SHORT', 'MEDIUM', 'LONG', 'FULL'.")

    def validate_isNameRequired(self):
        if 'isNameRequired' in self.params and not isinstance(self.params['isNameRequired'], bool):
            self.errors.append("isNameRequired must be boolean.")

    def validate_tag(self):
        if 'tag' in self.params and not isinstance(self.params['tag'], str):
            self.errors.append("tag must be a string.")

    def validate_operator(self):
        if 'operator' in self.params and self.params['operator'] not in ['AND', 'OR']:
            self.errors.append("operator must be one of 'AND', 'OR'.")

    def validate_charset(self):
        if 'charset' in self.params and self.params['charset'] != "UTF8":
            self.errors.append("charset must be UTF-8.")

    def validate_fuzzy(self):
        if 'fuzzy' in self.params and not _is_float(self.params['fuzzy']):
            self.errors.append("fuzzy must be a float between 0 and 1.")

    def validate_bounding_box(self):
        bbox_params = ['east', 'west', 'north', 'south']
        if any(param in self.params for param in bbox_params):
            for param in bbox_params:
                if param in self.params and not _is_float(self.params[param]):
                    self.errors.append(f"{param} must be a float.")

    def validate_orderby(self):
        if 'orderby' in self.params and self.params['orderby'] not in ['population', 'elevation', 'relevance']:
            self.errors.append("orderby must be one of 'population', 'elevation', 'relevance'.")

    def validate_inclBbox(self):
        if 'inclBbox' in self.params and self.params['inclBbox'] != 'true':
            self.errors.append("inclBbox can only be true.")

    def validate_username(self):
        if 'username' in self.params:
            self.errors.append("username shall not be provided.")


class ArticleSyntaxValidator:
    def __init__(self):
        self.geonames_syntax_validator = SearchParameterSyntaxValidator()

    def validate_toponyms_of_article(self, llm_output: LLMOutput) -> ValidatedOutput:
        try:
            validated_output = ValidatedOutput(**llm_output.model_dump())

            # needed in baseline generation chain, not in agent graph
            if validated_output.fatal_errors:
                return validated_output

            temp_gt_toponyms = [toponym.casefold() for toponym in validated_output.toponyms]

            in_reflection_phase = False
            if hasattr(validated_output, 'reflection_phase'):  # needed in candidate generation chain, not in agent graph
                if validated_output.reflection_phase == ReflectionPhase.ACTOR_RETRY_ON_INVALID_TOPONYMS:
                    in_reflection_phase = True
                    for toponym_with_search_params in validated_output.parsed_output:
                        toponym_with_search_params.generated_by_retry = True
                    already_correct_toponyms = [topo.toponym.casefold() for topo in itertools.chain(
                        validated_output.valid_toponyms, validated_output.duplicate_toponyms)]
                    temp_gt_toponyms = temp_gt_toponyms - already_correct_toponyms

            # First, all parsed toponyms can either have valid syntax, invalid syntax, or be duplicates
            for toponym_with_search_params in llm_output.parsed_output:
                if toponym_with_search_params.params:
                    is_valid, errors = self.geonames_syntax_validator.validate(toponym_with_search_params.params)
                    if is_valid:
                        validated_output.valid_toponyms.append(toponym_with_search_params)
                    else:
                        validated_output.invalid_toponyms.append(ToponymSearchArgumentsWithErrors(
                            **toponym_with_search_params.model_dump(),
                            errors_per_toponym=[
                                Error(
                                    execution_step=ExecutionStep.SEARCHPARAMETERVALIDATOR,
                                    error_message=str(errors)
                                )
                            ]
                        ))
                elif toponym_with_search_params.duplicate_of:
                    validated_output.duplicate_toponyms.append(toponym_with_search_params)
                else:
                    validated_output.invalid_toponyms.append(ToponymSearchArgumentsWithErrors(
                        toponym=toponym_with_search_params.toponym,
                        errors_per_toponym=[
                            Error(
                                execution_step=ExecutionStep.SEARCHPARAMETERVALIDATOR,
                                error_message="None of search parameter or duplicate key provided."
                            )
                        ]
                    ))

            # Second, we need to check if the toponyms with valid syntax are actually in the article (both for valid
            # and duplicate toponyms) create local copies of lists to avoid changing the original lists and allow
            # stable iteration

            if not in_reflection_phase:
                valid_toponyms = validated_output.valid_toponyms.copy()
                duplicate_toponyms = validated_output.duplicate_toponyms.copy()
            else:
                valid_toponyms = [valid for valid in validated_output.valid_toponyms if valid.generated_by_retry]
                duplicate_toponyms = [duplicate for duplicate in validated_output.duplicate_toponyms if duplicate.generated_by_retry]

            for generation in itertools.chain(valid_toponyms, duplicate_toponyms):
                if generation.toponym.casefold() in temp_gt_toponyms:
                    temp_gt_toponyms.remove(generation.toponym.casefold())
                else:
                    validated_output.invalid_toponyms.append(ToponymSearchArgumentsWithErrors(
                        **generation.model_dump(),
                        errors_per_toponym=[
                            Error(
                                execution_step=ExecutionStep.SEARCHPARAMETERVALIDATOR,
                                error_message="Generation does not reference a toponym in the article."
                            )
                        ]
                    ))
                    if generation in validated_output.valid_toponyms:
                        validated_output.valid_toponyms.remove(generation)
                    else:
                        validated_output.duplicate_toponyms.remove(generation)

            # Third, we need to check if all duplicates reference a valid toponym
            duplicate_toponyms = validated_output.duplicate_toponyms.copy()
            for duplicate in duplicate_toponyms:
                valid_duplicate = False
                for valid_toponym in validated_output.valid_toponyms:  # harsh because duplicate could also be correctly referring to an invalid toponym
                    if duplicate.duplicate_of.casefold() == valid_toponym.toponym.casefold():
                        valid_duplicate = True
                        break
                if not valid_duplicate:  # didn't find a valid toponym to reference for the duplicate
                    validated_output.invalid_toponyms.append(ToponymSearchArgumentsWithErrors(
                        **duplicate.model_dump(),
                        errors_per_toponym=[
                            Error(
                                execution_step=ExecutionStep.SEARCHPARAMETERVALIDATOR,
                                error_message="Duplicate toponym does not reference a valid generated toponym."
                            )
                        ]
                    ))
                    validated_output.duplicate_toponyms.remove(duplicate)

            # the sum of all generated toponyms
            temp_toponym_list = [temp_toponym.toponym.casefold() for temp_toponym in (validated_output.valid_toponyms +
                                                                                      validated_output.duplicate_toponyms +
                                                                                      validated_output.invalid_toponyms)]
            if len(temp_toponym_list) < len(validated_output.toponyms):  # too few toponyms generated
                for mention in validated_output.toponyms:
                    if mention.casefold() not in temp_toponym_list:
                        validated_output.invalid_toponyms.append(ToponymSearchArgumentsWithErrors(
                            toponym=mention,
                            errors_per_toponym=[
                                Error(
                                    execution_step=ExecutionStep.SEARCHPARAMETERVALIDATOR,
                                    error_message="No valid search arguments generated."
                                )
                            ]
                        ))
                    else:
                        temp_toponym_list.remove(mention.casefold())


            # just for code validation
            nof_valid_invalid_duplicate = len([temp_toponym.toponym for temp_toponym in (validated_output.valid_toponyms +
                                                                                   validated_output.duplicate_toponyms +
                                                                                   validated_output.invalid_toponyms)])

            nof_valid_duplicate = len([temp_toponym.toponym for temp_toponym in (validated_output.valid_toponyms +
                                                                            validated_output.duplicate_toponyms)])

            correct_nof_toponyms = len(validated_output.toponyms)
            if nof_valid_invalid_duplicate < correct_nof_toponyms:  # indicates most probably a coding error
                print("FATAL ERROR: TOO FEW TOPONYMS: VALIDATION ERROR for article ", validated_output.article_id)
            if nof_valid_duplicate > correct_nof_toponyms: # indicates most probably a coding error
                print("FATAL ERROR: TOO MANY TOPONYMS: VALIDATION ERROR for article ", validated_output.article_id)

            return validated_output
        except Exception as e:
            validated_output.fatal_errors = [Error(execution_step=ExecutionStep.ARTICLESYNTAXVALIDATOR,
                                                   error_message=str(e))]
            return validated_output


# test case
if __name__ == "__main__":
    article_ids = ['44148889', '44228209']
    load_dotenv()
    working_memory = WorkingMemory()
    prompt = working_memory.create_final_prompt()
    handler = ChatAIHandler()
    llm = handler.get_model("meta-llama-3.1-8b-instruct")
    for article_id in article_ids:
        article = working_memory.few_shot_handler.data_handler.get_article_for_prompting(article_id)
        toponyms = working_memory.few_shot_handler.data_handler.get_short_toponyms_for_article(article_id)
        tops = str(toponyms)
        parser = OutputParser(article_id=article_id, toponym_list=toponyms)
        validator = ArticleSyntaxValidator()
        chain = prompt | llm | parser.extract_output | validator.validate_toponyms_of_article
        llm_answer = chain.invoke(
            {
                "input__heading": article.get('title'),
                "input__news_article": article.get('text'),
                "input__toponym_list": tops
            }
        )
        print(llm_answer)
