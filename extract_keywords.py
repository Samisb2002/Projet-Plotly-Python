import json

class ExtractKeywords:

    @staticmethod
    def extract_keywords(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data['metadata']['all']['kws']
