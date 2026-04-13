import os
import numpy as np
from importlib_resources import files
from amn import *
from amn.Build_Model import Neural_Model, model_input
from amn.Build_Dataset import TrainingSet, get_index_from_id
from amn.Build_Model import evaluate_model, train_evaluate_model
from amn.Import import *
from sklearn.metrics import r2_score


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


def build_training_set(
    organism="custom",

    # File & data overrides
    medium_file=None,
    cobra_file=None,

    # Training params overrides
    method = 'EXP',  # FBA, pFBA or EXP
    #reduce = False,  # Set at True if you want to reduce the model
    seed=10,
    reduce=None

    # output dir
    #trainingfile=None
):
    np.random.seed(seed=seed) 
    # Get parameters from medium file
    #cobrafile = f'{DIRECTORY}{cobraname}'
    #mediumfile = f'{DIRECTORY}{mediumname}'

    # DEFAULT PRESETS BY ORGANISM
    PRESETS = {
        "putida": {
            "cobra_file": get_default_model('putida','build')
        },

        "ecoli": {
            "cobra_file": get_default_model('ecoli','build')
        },

        "biolog": {
            "cobra_file": get_default_model('biolog','build')
        }
    }
    #if organism not in PRESETS:
        #raise ValueError(f"Organism must be one of {list(PRESETS.keys())}")

    cfg = PRESETS[organism]

    # Apply overrides if provided
    cobra_file = cobra_file if cobra_file is not None else cfg["cobra_file"]
    
    #cadapt with the library codes
    cobra_name = cobra_file.replace('.xml','')
    medium_name = medium_file.replace('.csv','')

    parameter = TrainingSet(cobraname=cobra_name,
                            mediumname=medium_name,
                            method=method,
                            verbose=False)
    
    return parameter

    #trainingfile = f'{DIRECTORY}{mediumname}'
    #parameter.save(trainingfile, reduce=reduce)

    # Verifying
    #parameter = TrainingSet()
    #parameter.load(trainingfile)
    #print(trainingfile)
    #parameter.printout()

def train_gr_prediction(
    organism="custom",
    
    # File & data overrides
    file_name=None,
    trainingfile = None,
    cobraname_override = None,

    # Training params overrides
    model_type = None,  
    seed = 1,
    n_hidden = None,
    hidden_dim = None,
    Maxloop = 3,
    epochs = None,
    xfold = None,
    niter = None,
    batch_size = None,
    objective = True

    # output dir
    #modul_dir = None
):
    np.random.seed(seed=seed)

    PRESETS = {
        "putida": {
            "trainingfile": get_default_model('putida',"gr"),
            "cobraname_override": get_default_model('putida',"build"),
            "file_name": "IJN1463EXP",
            "n_hidden": 1,
            "hidden_dim": 500,
            "epochs": 500,
            "xfold": 5,
            "niter": 0,
            "batch_size": 10
        },

        "ecoli": {
            "trainingfile": get_default_model('ecoli',"gr"),
            "cobraname_override": get_default_model('ecoli',"build"),
            "n_hidden": 1,
            "hidden_dim": 1000,
            "epochs": 1000,
            "xfold": 5,
            "niter": 0,
            "batch_size": 10
        },

        "biolog": {
            "trainingfile": get_default_model('biolog',"gr"),
            "cobraname_override": get_default_model('biolog',"build"),
            "n_hidden": 1,
            "hidden_dim": 1000,
            "epochs": 1000,
            "xfold": 5,
            "niter": 0,
            "batch_size": 100
        }
    }
    #if organism not in PRESETS:
        #raise ValueError(f"Organism must be one of {list(PRESETS.keys())}")

    cfg = PRESETS[organism]

    # Apply overrides if provided
    trainingfile = trainingfile if trainingfile is not None else cfg["trainingfile"]
    n_hidden = n_hidden if n_hidden is not None else cfg["n_hidden"]
    hidden_dim = hidden_dim if hidden_dim is not None else cfg["hidden_dim"]
    epochs = epochs if epochs is not None else cfg["epochs"]
    xfold = xfold if xfold is not None else cfg["xfold"]
    niter = niter if niter is not None else cfg["niter"]
    batch_size = batch_size if batch_size is not None else cfg["batch_size"]
    cobraname_override = cobraname_override if cobraname_override is not None else cfg["cobraname_override"]

    #cadapt with the library codes
    trainingfile_name = trainingfile.replace('.npz','')
    cobraname_override_name = cobraname_override.replace('.xml','') if cobraname_override is not None else cobraname_override.replace('.xml','')

    reservoirname = f'{cobraname_override_name}_{model_type}'
    #reservoirfile = f'{modul_dir}{reservoirname}'
    # Training
    for Nloop in range(Maxloop):
        
        model = Neural_Model(trainingfile=trainingfile_name,
                model_type=model_type,
                scaler=True,
                objective=objective,
                n_hidden=n_hidden, hidden_dim=hidden_dim, output_dim=1,
                activation='relu', # activation for last layer
                scoring_function=r2_score, 
                epochs=epochs, xfold=xfold, niter=niter, batch_size=batch_size,
                verbose=False,
                cobraname_override=cobraname_override_name)
        model.printout()
        
        # Train and evaluate
        start_time = time.time()
        reservoir, pred, stats, _ = train_evaluate_model(model, verbose=True)
        delta_time = time.time() - start_time

        # Printing cross-validation results
        stats.printout(reservoirname, delta_time)
        r2 = r2_score(model.Y[:,0], pred[:,0], multioutput='variance_weighted')
        print(f'Iter {Nloop} Collated R2 {r2:.4f}')
        R2=[]
        R2.append(r2)
        PRED=[]
        PRED.append(pred[:, 0])
        #if r2 == max(R2):  # save the best model
            #reservoir.save(reservoirfile)

    return r2, R2, PRED, reservoirname, reservoir, stats, model, trainingfile, cobraname_override
        
    # Some printing
    #R2, PRED = np.asarray(R2), np.asarray(PRED)
    #print(f'{trainname} Averaged R2 = {np.mean(R2):4f} ± {np.std(R2):.4f} Best R2 = {np.max(R2):.4f}')
    #reservoir.load(reservoirfile, output_dim=1)
    #reservoir.printout()
    #X, Y = model_input(reservoir, verbose=False)
    #print(X.shape, Y.shape)
    #pred, _ = evaluate_model(reservoir.model, X, Y, reservoir, verbose=False)
    #y = pred[:,:model.Y.shape[1]]     
    #R2 = r2_score(model.Y[:,0], y[:,0], multioutput='variance_weighted')
    #print(f'Final R2 {R2:.4f}')
