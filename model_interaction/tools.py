import os
import re
import time
import urllib.parse
import pycountry
import requests
from dotenv import load_dotenv


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


class GeoNamesSyntaxValidation:
    def __init__(self, params):
        self.params = params
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

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the syntax of the GeoNames API parameters
        :return:
        """
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


class GeoNamesAPI:
    def __init__(self, params):
        self.params = params
        self.base_url = "http://api.geonames.org/search?"

    def search(self):
        self.params.update({'username': os.getenv('GEONAMES_USERNAME')})
        url = self.base_url + urllib.parse.urlencode(self.params)
        response = requests.get(url)
        return response.json()


# test case
if __name__ == "__main__":
    load_dotenv()
    params = {
        'q': 'Somis ventura county',
        'maxRows': '10',
        'type': 'json',
        'style': 'FULL',
        'isNameRequired': True,
        'orderby': 'relevance',
        'inclBbox': 'true'
    }
    start_time = time.time()

    validation = GeoNamesSyntaxValidation(params)
    is_valid, errors = validation.validate()
    if is_valid:
        api = GeoNamesAPI(params)
        response = api.search()
        print(response)
    else:
        print(errors)
    print(f"Time taken: {time.time() - start_time:.2f} seconds")