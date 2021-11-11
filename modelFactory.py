from models.ColorMoments import ColorMoments
from models.ELBP import ELBP
from models.HOG import HOG

cm = ColorMoments()
elbp = ELBP()
hog = HOG()
models = [cm, elbp, hog]

# returns a model for the given name
def get_model(modelName):
    modelName = modelName.lower()
    if modelName == 'cm':
        return cm
    elif modelName == 'elbp':
        return elbp
    elif modelName == 'hog':
        return hog
    else:
        print("No such model exists")

# returns all the models
def get_all_models():
    return models
