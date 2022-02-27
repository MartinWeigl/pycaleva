"""
This module holds various metrics in context of calibration measurement of binary probabilistic classfication.
"""

import numpy as np


def brier(y,p,normalize=False):
    """Calculate the brier score for given binary class labels and the according class probabilities.
            
        Parameters
        ----------
        y : array_like
                Expected class labels given in test set. (Ground truth y)
        p : array_like
                Observed probabilities predicted by a classification model.
        normalize : bool, optional
                Decides if brier score should be normalized to range [0,1].
                Defaults to False.
                
        Returns
        -------
        b : float
            The Brier score for given data.
        
        Notes
        -----
        This score can be used for validation of classification models representing both, the discrimination and calibration
        of the model. A low score is indicating good discrimination and calibration in context of given test data.
        
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

    if normalize:
        p_mean = p.mean()
        return (brier_score) / ( p_mean * (1 - p_mean) )

    return (np.square(y-p)).sum() / len(y)

    

def brier_skill_score(y,p):
    """Calculate the brier skill score for given binary class labels and the according class probabilities.
            
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
        of the model. A score of 1.0 would mean perfect discrimination and calibration in context of given test data. 
        A low score would indicate poor discrimination and calibration.
        
        References
        ----------
        ..  [1] Huang, Y., Li, W., Macheret, F., Gabriel, R. A., & Ohno-Machado, L. (2020). 
            A tutorial on calibration measurements and calibration models for clinical prediction models. 
            Journal of the American Medical Informatics Association, 27(4), 621-633.
            [2] Steyerberg, E. W., Vickers, A. J., Cook, N. R., Gerds, T., Gonen, M., Obuchowski, N., ... & Kattan, M. W. (2010). 
            Assessing the performance of prediction models: a framework for some traditional and novel measures. 
            Epidemiology (Cambridge, Mass.), 21(1), 128.
        """

    p_mean = p.mean()
    brier_score = brier(y, p)

    return ( 1 - (brier_score) / ( p_mean * (1 - p_mean) ) )


def __tpr_fpr(y_thresh, y_test):
    true_positive = np.equal(y_thresh, 1) & np.equal(y_test, 1)
    true_negative = np.equal(y_thresh, 0) & np.equal(y_test, 0)
    false_positive = np.equal(y_thresh, 1) & np.equal(y_test, 0)
    false_negative = np.equal(y_thresh, 0) & np.equal(y_test, 1)

    tpr = true_positive.sum() / (true_positive.sum() + false_negative.sum())
    fpr = false_positive.sum() / (false_positive.sum() + true_negative.sum())

    return tpr, fpr

def __roc(pred, y_test, partitions):
    roc = np.array([])
    for i in range(partitions + 1):
        y_thresh = np.greater_equal(pred, i / partitions).astype(int)
        tpr, fpr = __tpr_fpr(y_thresh, y_test)
        roc = np.append(roc, [fpr, tpr])
    return roc.reshape(-1, 2)


def auroc(y, p):
    if len(np.unique(y)) != 2:
        raise ValueError(
            "Only one class label present in data. AUROC is undefined in that case."
        )

    partitions = 250
    ROC = __roc(p, y, partitions=partitions)
    fpr, tpr = ROC[:, 0], ROC[:, 1]
    auroc = 0
    for k in range(partitions):
        auroc = auroc + (fpr[k]- fpr[k + 1]) * tpr[k]
    return auroc