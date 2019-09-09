import sklearn
import os
from os.path import join
import pickle

class ClassifModel(object):

    def __init__(self):
        model_filename = join(os.path.dirname(__file__), "model_files/rforest_explorers.sav")
        self.model = load_lr_model = pickle.load(open(model_filename, 'rb'))

        self.labels_list = ['Animals-Others', 'Animals-air', 'Animals-earthly',
       'Animals-submarine', 'Culture&Tradition-ArcheologyandHouse',
       'Culture&Tradition-Cooking', 'Culture&Tradition-Craft',
       'Culture&Tradition-Education', 'Culture&Tradition-Health',
       'Culture&Tradition-Music_and_Dance', 'Culture&Tradition-Other',
       'Culture&Tradition-UrbanArt', 'Landscape-Astronomy',
       'Landscape-Coastline', 'Landscape-Moutain', 'Landscape-Vulcanos',
       'Landscapes-Desert', 'Landscapes-Jungle', 'Landscapes-Other',
       'Landscapes-Plain', 'Landscapes-Urban', 'Landscapes-coursdeau',
       'Other-Other', 'Vegetal_Flowers', 'Vegetal_Other',
       'Vegetal_Plants', 'Vegetal_Trees']

    def get_label(self, tags_scores):

        X = tags_scores.round(4)
        label = self.model.predict([X])

        prediction = self.labels_list[label[0]]

        return prediction