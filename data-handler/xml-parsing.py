import xml.etree.ElementTree as ET
import pandas as pd
import os

class XMLDataHandler:
    def __init__(self, data_directory):
        self.data_directory = data_directory
        self.articles_df = pd.DataFrame()
        self.toponyms_with_gaztag_df = pd.DataFrame()
        self.toponyms_without_gaztag_df = pd.DataFrame()

    def parse_xml(self, file_name):
        file_path = os.path.join(self.data_directory, file_name)
        tree = ET.parse(file_path)
        root = tree.getroot()

        articles_data = []
        toponyms_with_gaztag_data = []
        toponyms_without_gaztag_data = []

        for article in root.findall('article'):
            article_data = self._extract_article_data(article)
            articles_data.append(article_data)

            for toponym in article.find('toponyms').findall('toponym'):
                toponym_data = self._extract_toponym_data(article, toponym)
                if toponym.find('gaztag') is not None:
                    toponyms_with_gaztag_data.append(toponym_data)
                else:
                    toponyms_without_gaztag_data.append(toponym_data)

        self.articles_df = pd.DataFrame(articles_data)
        self.toponyms_with_gaztag_df = pd.DataFrame(toponyms_with_gaztag_data)
        self.toponyms_without_gaztag_df = pd.DataFrame(toponyms_without_gaztag_data)

    def _extract_article_data(self, article):
        return {
            'docid': article.get('docid'),
            'feedid': article.find('feedid').text,
            'title': article.find('title').text,
            'domain': article.find('domain').text,
            'url': article.find('url').text,
            'dltime': article.find('dltime').text,
            'text': article.find('text').text
        }

    def _extract_toponym_data(self, article, toponym):
        toponym_data = {
            'docid': article.get('docid'),
            'start': toponym.find('start').text,
            'end': toponym.find('end').text,
            'phrase': toponym.find('phrase').text,
        }

        gaztag = toponym.find('gaztag')
        if gaztag is not None:
            toponym_data.update(self._extract_gaztag_data(gaztag))

        return toponym_data

    def _extract_gaztag_data(self, gaztag):
        return {
            'geonameid': gaztag.get('geonameid'),
            'name': gaztag.find('name').text if gaztag.find('name') is not None else None,
            'fclass': gaztag.find('fclass').text if gaztag.find('fclass') is not None else None,
            'fcode': gaztag.find('fcode').text if gaztag.find('fcode') is not None else None,
            'lat': gaztag.find('lat').text if gaztag.find('lat') is not None else None,
            'lon': gaztag.find('lon').text if gaztag.find('lon') is not None else None,
            'country': gaztag.find('country').text if gaztag.find('country') is not None else None,
            'admin1': gaztag.find('admin1').text if gaztag.find('admin1') is not None else None
        }

    def get_toponyms_for_article(self, docid):
        return self.toponyms_with_gaztag_df[self.toponyms_with_gaztag_df['docid'] == docid]

    def get_toponyms_without_gaztag(self):
        return self.toponyms_without_gaztag_df

    def find_all_fields(self, file_name):
        file_path = os.path.join(self.data_directory, file_name)
        tree = ET.parse(file_path)
        root = tree.getroot()
        fields = set()

        def traverse(element):
            fields.add(element.tag)
            for child in element:
                traverse(child)

        traverse(root)
        return fields

    def count_duplicate_toponyms(self):
        # Check if the DataFrame is empty
        if self.toponyms_with_gaztag_df.empty:
            return 0

        # Group by article and count duplicates within each article
        duplicate_counts = self.toponyms_with_gaztag_df.groupby('docid').apply(
            lambda x: x['phrase'].duplicated().sum()
        )

        # Sum the counts of duplicates across all articles
        total_duplicates = duplicate_counts.sum()

        return total_duplicates


# Example usage
if __name__ == "__main__":
    data_handler = XMLDataHandler('data/')
    data_handler.parse_xml('LGL_test.xml')
    article_docid = '39423136'
    toponyms_for_article = data_handler.get_toponyms_for_article(article_docid)
    print("Toponyms for article:", toponyms_for_article)
    print("Number of words in the text of all articles:", data_handler.articles_df['text'].apply(lambda x: len(x.split())).sum())
    # count how many toponyms are duplicate in an article for all articles
    print("Number of duplicate toponyms:", data_handler.count_duplicate_toponyms())

