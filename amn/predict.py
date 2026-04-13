import numpy as np
from importlib_resources import files
from amn.Import import *
from amn.Build_Model import Neural_Model, model_input
from amn.Build_Model import evaluate_model, train_evaluate_model
from amn.Utilities import read_XY
from sklearn.metrics import r2_score, accuracy_score


def get_default_model(organism, type):
    if organism.lower() == "putida" and type.lower() == "build":
        return str(files("amn.models").joinpath("IJN1463EXP.xml"))
    elif organism.lower() == "putida" and type.lower() == "gr":
        return str(files("amn.models").joinpath("IJN1463EXP.npz"))
    elif organism.lower() == "ecoli" and type.lower() == "build":
        return str(files("amn.models").joinpath("iML1515EXP.xml"))
    elif organism.lower() == "ecoli" and type.lower() == "gr":
        return str(files("amn.models").joinpath("iML1515EXP.npz"))
    elif organism.lower() == "biolog" and type.lower() == "build":
        return str(files("amn.models").joinpath("BIOLOGEXP.xml"))
    elif organism.lower() == "biolog" and type.lower() == "gr":
        return str(files("amn.models").joinpath("BIOLOGEXP.npz"))
    else:
        raise ValueError("Custom model must be provided")
    
def predict_amn(
    organism="putida",

    # INPUT folders
    modelfile=None, # .h5
    trainname=None, # .npz
    fileparam=None, # .csv
    cobra_file=None, # .xml

    # Evaluation params overrides
    seed = 1,
    model_type = 'AMN_QP',
    metric=None,
    n_hidden=1,
    hidden_dim=1000,
    objective = True,

    #OUTPUT folder
    predname="toto"

):
    np.random.seed(seed=seed)
    # DEFAULT PRESETS BY ORGANISM
    PRESETS = {
        "putida": {
            "cobra_file": get_default_model('putida','build'),
            "metric": "accuracy_score"
        },

        "ecoli": {
            "cobra_file": get_default_model('ecoli','build'),
            "metric": "r2_score"
        },

        "biolog": {
            "cobra_file": get_default_model('biolog','build'),
            "metric": "r2_score"
        }
    }

    #if organism not in PRESETS:
        #raise ValueError(f"Organism must be one of {list(PRESETS.keys())}")

    cfg = PRESETS[organism]

    # Apply overrides if provided
    cobra_file = cobra_file if cobra_file is not None else cfg["cobra_file"]
    if organism == "putida":
        metric="accuracy_score"
    else:
        metric = metric


    cobra_file = cobra_file.replace('.xml','')
    trainname = trainname.replace('.npz','')

    model = Neural_Model(
        trainingfile=trainname,
        cobraname_override=cobra_file)
    model.load(filename="", fileparam=fileparam, filemodel=modelfile, output_dim=1)
    #model.printout()
    #H, X, _ = read_XY(predfile, nY=0, scaling='')
    #pred, _ = evaluate_model(model.model, X, [], model, verbose=False)
    #y_pred = pred[:,0]  if metric == r2_score else pred[:,0].round()
    #print(y_pred[0:10])
    #R2 = metric(Y[:,0], y_pred)
    #print(f'Final Metric {R2:.4f}')

    return model, metric