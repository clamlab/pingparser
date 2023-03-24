"""
data containers for bonsai event ping data
"""


class Experiment:
    def __init__(self):
        self.anim = {} #container for AnimalData
        self.notes = {} #make note of any significant events e.g. food restriction
        self.data_root = ''


class AnimalData:
    def __init__(self, name):
        self.name = name
        self.subsess_paths = {}
        self.subsess = {}


