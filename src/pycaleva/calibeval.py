
""" This module holds the class *CalibrationEvaluator* that bundles functionalities of the PyCalEva Framework."""

from ._basecalib import _BaseCalibrationEvaluator
from ._report import _Report

# This class extends the _BaseCalibrationEvaluator adding a pdf-Report functionality
class CalibrationEvaluator(_BaseCalibrationEvaluator):

    def calibration_report(self, filepath:str, model_name:str) -> None:
        """Create a pdf-Report including statistical tests and plots regarding the calibration of a model.

        Parameters
        ----------
            filepath: str
                        The filepath for the output file.
            model_name: str
                        The name for the evaluated model.
        """
        _Report().create(filepath, model_name, self)