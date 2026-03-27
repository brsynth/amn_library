import os
import zipfile
import tempfile
import argparse
import shutil
import numpy as np
from contextlib import redirect_stdout
from sklearn.metrics import r2_score
from amn.Build_Dataset import TrainingSet, get_index_from_id
from amn.train import *
from amn.predict import *

def train_build(argv=None):
    parser = argparse.ArgumentParser(description="AMN Set Trainer",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--organism", choices=["ecoli", "putida", "biolog", "custom"], default="custom",
                        help="model")

    parser.add_argument("--medium-file", type=str,
                        help="Medium file")
    parser.add_argument("--cobra-file", type=str,
                        help="COBRA file")
    parser.add_argument("--method", choices=["FBA", "pFBA", "EXP"], default="EXP", type=str,
                        help="Method")

    parser.add_argument("--reduce", type=lambda x: x.lower() == "true", default=False,
                        help="Set at True if you want to reduce the model")
    parser.add_argument("--seed", type=int,
                        help="Random seed")
    
    parser.add_argument("--modul-dir",type=str,default=os.path.join(os.getcwd(), "modul"),
        help="Directory to save outputs (default: ./modul in working directory)") 

    args = parser.parse_args(argv)

    # create directiries
    trainingfile = args.modul_dir
    os.makedirs(trainingfile, exist_ok=True)

    (
        parameter
    ) = build_training_set(
        organism=args.organism,
        medium_file=args.medium_file,
        cobra_file=args.cobra_file,
        method=args.method,
        reduce=args.reduce,
        seed=args.seed
    )

    parameter.save(trainingfile, reduce=args.reduce)
    parameter = TrainingSet()
    parameter.load(trainingfile)

    output_txt = os.path.join(trainingfile, "training_results.txt")
    # Write everything into the txt file
    with open(output_txt, "w") as f:
        f.write(f"Training directory: {trainingfile}\n\n")
        
        # Redirect printout() output into file
        from contextlib import redirect_stdout
        with redirect_stdout(f):
            parameter.printout()

    return output_txt

def train_gr (argv=None):
    parser = argparse.ArgumentParser(description="AMN Set Trainer",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--organism", choices=["ecoli", "putida", "biolog","custom"], default="custom",
                        help="model")

    parser.add_argument("--file-name", type=str,
                        help="Model file name")
    parser.add_argument("--trainingfile", type=str,
                        help="Training file")
    parser.add_argument("--cobraname-override", type=str,
                        help="Training file")

    parser.add_argument("--model-type", type=str, default="AMN_QP",
                        help="Type Of Model")
    parser.add_argument("--n-hidden", type=int, 
                        help="N Hidden")
    parser.add_argument("--hidden-dim", type=int, 
                        help="Hidden dimention")
    parser.add_argument("--maxloop", type=int, default=3,
                        help="Loop Max")
    parser.add_argument("--epochs", type=int, 
                        help="epochs")
    parser.add_argument("--xfold", type=int, 
                        help="Number of folds for cross-validation")
    parser.add_argument("--niter", type=int,
                        help="Number of training iterations")
    parser.add_argument("--batch-size", type=int, 
                        help="Training batch size")
    parser.add_argument("--seed", type=int,
                        help="Random seed")
    
    parser.add_argument("--modul-dir",type=str,default=os.path.join(os.getcwd(), "modul"),
        help="Directory to save outputs (default: ./modul in working directory)") 

    args = parser.parse_args(argv)

    # create directiries
    modul_dir = args.modul_dir
    os.makedirs(modul_dir, exist_ok=True)
 

    (
        r2,
        R2,
        PRED,
        reservoirname,
        reservoir,
        stats,
        model,
        trainingfile,
        cobraname_override
    ) = train_gr_prediction(
        organism=args.organism,
        trainingfile=args.trainingfile,
        model_type=args.model_type,
        n_hidden=args.n_hidden,
        hidden_dim=args.hidden_dim,
        Maxloop=args.maxloop,
        epochs=args.epochs,
        xfold=args.xfold,
        niter=args.niter,
        batch_size=args.batch_size,
        seed=args.seed,
        cobraname_override=args.cobraname_override
    )
    reservoirname=os.path.basename(reservoirname)
    reservoirfile = f'{modul_dir}/{reservoirname}'

    if r2 == max(R2):  # save the best model
            reservoir.save(os.path.join(reservoirfile))
    
    shutil.copy(trainingfile, modul_dir)
    shutil.copy(cobraname_override, modul_dir)

    R2, PRED = np.asarray(R2), np.asarray(PRED)
    output_txt = os.path.join(modul_dir, f"{args.file_name}_training_summary.txt")

    with open(output_txt, "w") as f:
        with redirect_stdout(f):
            print(f"{args.file_name} Averaged R2 = {np.mean(R2):.4f} ± {np.std(R2):.4f} Best R2 = {np.max(R2):.4f}\n")

            reservoir.load(reservoirfile, output_dim=1)
            reservoir.printout()

            X, Y = model_input(reservoir, verbose=False)
            print(f"X shape: {X.shape}, Y shape: {Y.shape}")

            pred, _ = evaluate_model(reservoir.model, X, Y, reservoir, verbose=False)
            y = pred[:, :model.Y.shape[1]]
            final_R2 = r2_score(model.Y[:,0], y[:,0], multioutput='variance_weighted')
            print(f"Final R2: {final_R2:.4f}")

     #Zip model
    zip_file_path = f"{modul_dir}.zip"
    shutil.make_archive(base_name=modul_dir, format='zip', root_dir=modul_dir)

    return output_txt, reservoirfile, zip_file_path

def pred_amn (argv=None):
    parser = argparse.ArgumentParser(description="AMN Prediction",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--organism", choices=["ecoli", "putida", "biolog","custom"], default="custom",
                        help="model")
    
    parser.add_argument("--zipInput", type=str, default=None,
                        help="contain _model.h5, _param.csv, .xml, .npz  (.zip)")

    parser.add_argument("--modelfile", type=str,
                        help="Model file name (.h5)")
    parser.add_argument("--trainingfile", type=str,
                        help="Training file (.npz)")
    parser.add_argument("--fileparam", type=str,
                        help="parameter file (.csv)")
    parser.add_argument("--predfile", type=str,
                        help="predict file Values (.csv)")
    parser.add_argument("--cobraname-override", type=str,
                        help="Training file (.xml)")
    parser.add_argument("--objective", type=lambda x: x.lower() == "true", default=True,
                        help="Objective")
    parser.add_argument("--model-type", type=str, default="AMN_QP",
                        help="Type Of Model")
    parser.add_argument("--n-hidden", type=int, 
                        help="N Hidden")
    parser.add_argument("--hidden-dim", type=int, 
                        help="Hidden dimention")
    parser.add_argument("--metric", choices=["accuracy_score","r2_score"], default="r2_score",
                        help="accuracy_score for P. putida only")
    parser.add_argument("--seed", type=int,
                        help="Random seed")
    
    parser.add_argument("--modul-dir",type=str,default=os.path.join(os.getcwd()),
        help="Directory to save outputs (default: in working directory)") 

    args = parser.parse_args(argv)

    # create directiries
    modul_dir = args.modul_dir
    os.makedirs(modul_dir, exist_ok=True)

    if args.zipInput:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(args.zipInput) as z:
                z.extractall(tmp_dir)
                extracted_train = [os.path.join(dp, f) 
                           for dp, dn, filenames in os.walk(tmp_dir) 
                           for f in filenames if f.endswith(".npz")]

                extracted_param = [os.path.join(dp, f) 
                                for dp, dn, filenames in os.walk(tmp_dir) 
                                for f in filenames if f.endswith("_param.csv")]

                extracted_model = [os.path.join(dp, f) 
                                for dp, dn, filenames in os.walk(tmp_dir) 
                                for f in filenames if f.endswith(".h5")]
                
                extracted_cobra = [os.path.join(dp, f) 
                                for dp, dn, filenames in os.walk(tmp_dir) 
                                for f in filenames if f.endswith(".xml")]

                if len(extracted_train) != 1:
                    raise FileNotFoundError(f"Expected exactly one .npz file, found {len(extracted_train)}")
                if len(extracted_param) != 1:
                    raise FileNotFoundError(f"Expected exactly one _param.csv file, found {len(extracted_param)}")
                if len(extracted_model) != 1:
                    raise FileNotFoundError(f"Expected exactly one .h5 file, found {len(extracted_model)}")
                if len(extracted_cobra) != 1:
                    raise FileNotFoundError(f"Expected exactly one .h5 file, found {len(extracted_cobra)}")


            # Call prediction using extracted files
            model, metric = predict_amn(
                organism=args.organism,
                modelfile=extracted_model[0],
                trainname=extracted_train[0],
                fileparam=extracted_param[0],
                model_type=args.model_type,
                n_hidden=args.n_hidden,
                hidden_dim=args.hidden_dim,
                metric=args.metric,
                seed=args.seed,
                cobra_file=extracted_cobra[0],
                objective=args.objective
            )
           
    else:
        (
            model,
            metric
        ) = predict_amn(
            organism=args.organism,
            modelfile=args.modelfile,
            trainname=args.trainingfile,
            fileparam=args.fileparam,
            model_type=args.model_type,
            n_hidden=args.n_hidden,
            hidden_dim=args.hidden_dim,
            metric=args.metric,
            seed=args.seed,
            cobra_file=args.cobraname_override,
            objective=args.objective
        )
    
    output_file = os.path.join(modul_dir, "prediction_results.txt")
    predfile=args.predfile
    predfile=predfile.replace('.csv','')
    metric_func = r2_score if metric== "r2_score" else accuracy_score

    with open(output_file, "w") as f:
        with redirect_stdout(f):
            model.printout()
            H, X, Y = read_XY(predfile, nY=1, scaling='')
            print('X, Y, shapes:', X.shape, Y.shape)
            pred, _ = evaluate_model(model.model, X, [], model, verbose=False)
            y_pred = pred[:,0] if metric_func == r2_score else pred[:,0].round()
            print(y_pred[0:10])
            R2 = metric_func(Y[:,0], y_pred)
            print(f'Final Metric {R2:.4f}')

    return output_file


def main():
    parser = argparse.ArgumentParser(prog="amn", description="AMN CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("train_build", help="Build training set")
    subparsers.add_parser("train_gr", help="Train AMN for growth rate prediction")
    subparsers.add_parser("predict", help="Prediction AMN")
    args = parser.parse_args(sys.argv[1:2])

    # Dispatch
    if args.command == "train_build":
        train_build(sys.argv[2:])
    elif args.command == "train_gr":
        train_gr(sys.argv[2:])
    elif args.command == "predict":
        pred_amn(sys.argv[2:])

if __name__ == "__main__":
    main()
