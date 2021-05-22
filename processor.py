import json
import glob


class PoetryDataset:
    def __init__(self):
        pass

    def get_tang(self):
        poetries = []
        filenames = glob.glob('./data/*tang*.json')

        for filename in filenames:
            with open(filename) as f:
                poetries.extend([''.join(poetry['paragraphs'])
                                 for poetry in json.load(f)])

        return poetries
