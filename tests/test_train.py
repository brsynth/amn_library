import os
import pytest
from amn.train import train_gr_prediction

TEST_FOLDER = os.path.dirname(__file__)
DATA_FOLDER = os.path.join(TEST_FOLDER, "data")
#OUTPUT_FOLDER = os.path.join(TEST_FOLDER, "output")
#MODEL_FOLDER = os.path.join(OUTPUT_FOLDER, "model")
#FIGURE_FOLDER = os.path.join(OUTPUT_FOLDER, "figure")

#@pytest.fixture(autouse=True)
#def clean_output():
#    """Delete output folder before each test."""
#    if os.path.exists(OUTPUT_FOLDER):
#        shutil.rmtree(OUTPUT_FOLDER)
#    os.makedirs(OUTPUT_FOLDER)
#    yield


@pytest.mark.parametrize(
    "organism,file_name,model_type",
    [
        (
            "putida",
            "IJN1463EXP",
            "AMN_QP"
        )
    ]
)
def test_train_damn_full(organism, file_name, model_type):
    """Full functional test for DAMN training pipeline with in-memory outputs."""

    # Run the training and capture all returned variables
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
        organism=organism,
        file_name=file_name,
        model_type=model_type
    )

    # --- Type checks ---
    assert isinstance(r2, float), "r2 should be a float"
    assert isinstance(R2, list), "R2 should be a list"
    assert isinstance(PRED, list), "PRED should be a list"
    assert isinstance(reservoirname, str), "reservoirname should be a string"

    # --- Content checks ---
    assert len(R2) > 0, "R2 list should not be empty"
    assert len(PRED) > 0, "PRED list should not be empty"

    # --- Shape checks ---
    assert len(PRED[0]) > 0, "Prediction array should not be empty"

    # --- Value checks ---
    assert -1 <= r2 <= 1, "r2 should be between -1 and 1"

    assert reservoir is not None
    assert stats is not None
    assert model is not None

