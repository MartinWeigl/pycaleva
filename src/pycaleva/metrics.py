"""
This module holds some metrics in context of calibration measurement of binary probabilistic classfication.
"""

import numpy as np


def brier(y,p,scale=False):
    r"""Calculate the brier score for given binary class labels and the according class probabilities.
    
        Parameters
        ----------
        y : array_like
                Expected class labels given in test set. (Ground truth y)
        p : array_like
                Observed probabilities predicted by a classification model.
                
        Returns
        -------
        b : float
            The Brier score for given data.
        
        Notes
        -----
        This score can be used for validation of classification models representing both, the discrimination and calibration
        of the model. A low score is indicating good discrimination and calibration in context of given test data.
        
        Formula:

        .. math::
            B=\frac{1}{N} \sum_{i=1}^{N}\left(y_{i}-\hat{p}_{i}\right)^{2}

        References
        ----------
        ..  [1] Huang, Y., Li, W., Macheret, F., Gabriel, R. A., & Ohno-Machado, L. (2020). 
            A tutorial on calibration measurements and calibration models for clinical prediction models. 
            Journal of the American Medical Informatics Association, 27(4), 621-633.
            [2] Steyerberg, E. W., Vickers, A. J., Cook, N. R., Gerds, T., Gonen, M., Obuchowski, N., ... & Kattan, M. W. (2010). 
            Assessing the performance of prediction models: a framework for some traditional and novel measures. 
            Epidemiology (Cambridge, Mass.), 21(1), 128.
        """
    
    brier_score = (np.square(y-p)).sum() / len(y)
    return (np.square(y-p)).sum() / len(y)