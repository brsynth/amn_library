import os
import sys
import subprocess
import pytest

TEST_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(TEST_FOLDER, "data")

@pytest.mark.parametrize(
    "organism,metric",
    [
        ("putida", "accuracy_score")
    ]
)

def test_main_execution_putida(organism, metric):
    modelfile = os.path.join(DATA_FOLDER, "IJN1463EXP_AMN_QP_model.h5")
    trainingfile = os.path.join(DATA_FOLDER, "IJN1463EXP.npz")
    fileparam = os.path.join(DATA_FOLDER, "IJN1463EXP_AMN_QP_param.csv")
    predfile = os.path.join(DATA_FOLDER, "IJN1463VAL.csv")

    modul_dir = os.path.join(TEST_FOLDER, "test_output")
    os.makedirs(modul_dir, exist_ok=True)

    # Run the script as a subprocess
    result = subprocess.run(
        [
            sys.executable, "-m", "amn", "predict",
            "--organism", organism,
            "--modelfile", modelfile,
            "--trainingfile", trainingfile,
            "--fileparam", fileparam,
            "--predfile", predfile,
            "--metric", metric,
            "--modul-dir", modul_dir
        ],
        capture_output=True,
        text=True,
        cwd=os.path.join(TEST_FOLDER, "..")
    )

    # Ensure the script ran without crashing
    assert result.returncode == 0, f"Script failed: {result.stderr}"

    # Read the prediction output file
    output_file = os.path.join(modul_dir, "prediction_results.txt")
    assert os.path.isfile(output_file), "Output file was not created!"

    with open(output_file, "r") as f:
        content = f.read()

    # Check that the final metric is printed in the file
    assert "Final Metric" in content, "No final metric printed in output file!"