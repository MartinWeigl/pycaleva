
""" This module holds the class *CalibrationEvaluator* that bundles all functionalities of the PyCalEva Framework."""

from ._basecalib import _BaseCalibrationEvaluator
from ._report import _Report

__all__ = ['CalibrationEvaluator']


# This class extends the _BaseCalibrationEvaluator adding a pdf-Report functionality
class CalibrationEvaluator(_BaseCalibrationEvaluator):
    # Docstring inherited from _BaseCalibrationEvaluator !!

    def calibration_report(self, filepath:str, model_name:str) -> None:
        """Create a pdf-report including statistical tests and plots regarding the calibration of a binary classification model.

        Parameters
        ----------
            filepath: str
                    The filepath for the output file. Must end with '.pdf'
            model_name: str
                    The name for the evaluated model.
        """
        _Report().create(filepath, model_name, self)