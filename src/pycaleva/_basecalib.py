"""
A framework for calibration assessment of binary classification models 
written in Python.

References
----------
[1] Hosmer Jr, David W., Stanley Lemeshow, and Rodney X. Sturdivant.
        Applied logistic regression. Vol. 398. John Wiley & Sons, 2013.

[2] Pigeon, Joseph G., and Joseph F. Heyse.
    An improved goodness of fit statistic for probability prediction models.
    Biometrical Journal: Journal of Mathematical Methods in Biosciences 41.1 (1999): 71-82.

[3] Spiegelhalter, D. J. (1986). Probabilistic prediction in patient management and clinical trials.
    Statistics in medicine, 5(5), 421-433.

[4] Huang, Y., Li, W., Macheret, F., Gabriel, R. A., & Ohno-Machado, L. (2020).
    A tutorial on calibration measurements and calibration models for clinical prediction models.
    Journal of the American Medical Informatics Association, 27(4), 621-633.

[5] Jr, F. E. H. (2021). rms: Regression modeling strategies (R package version
    6.2-0) [Computer software]. The Comprehensive R Archive Network.
    Available from https://CRAN.R-project.org/package=rms

[6] Nattino, G., Finazzi, S., & Bertolini, G. (2014). A new calibration test 
    and a reappraisal of the calibration belt for the assessment of prediction models 
    based on dichotomous outcomes. Statistics in medicine, 33(14), 2390-2407.

[7] Bulgarelli, L. (2021). calibrattion-belt: Assessment of calibration in binomial prediction models [Computer software].
    Available from https://github.com/fabiankueppers/calibration-framework

[8] Nattino, G., Finazzi, S., Bertolini, G., Rossi, C., & Carrara, G. (2017).
    givitiR: The giviti calibration test and belt (R package version 1.3) [Computer
    software]. The Comprehensive R Archive Network.
    Available from https://CRAN.R-project.org/package=givitiR

[9] [Sturges, H. A. (1926). The choice of a class interval. 
    Journal of the american statistical association, 21(153), 65-66.]

[10] "Hosmer-Lemeshow test", https://en.wikipedia.org/wiki/Hosmer-Lemeshow_test

[11] Pigeon, Joseph G., and Joseph F. Heyse. "A cautionary note about assessing 
     the fit of logistic regression models." (1999): 847-853.
"""

from enum import Flag
from math import log2, ceil, sqrt
from typing import Union
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm
from scipy import integrate
from sklearn.metrics import roc_auc_score
from statsmodels.nonparametric.smoothers_lowess import lowess
from IPython.display import display

from .calbelt import CalibrationBelt
from .metrics import brier
from ._result_types import *


# SETS THE LIMIT FOR THE FREQUENCY IN CONTINGENCY TABLES
CHI_SQUARE_VIOLATION_LIMIT = 1

# SETS THE LIMIT FOR HIGHLIGHTING HIGH DEVIATION OF OBSERVED AND EXPECTED COUNTS IN CT
HIGHLIGHT_DEVIATION_LIMIT = 0.1

# SETS THE OUTPUT PRECISION OF FLOATS IN DATAFRAME
pd.options.display.precision = 3


class DEVEL(Flag):
    INTERNAL = False
    EXTERNAL = True


# _BaseCalibrationEvaluator  <--(inherits from)-- CalibrationEvaluator
class _BaseCalibrationEvaluator:

    # CONSTRUCTOR
    def __init__(self, y_true:np.ndarray, y_pred:np.ndarray, outsample:bool, n_groups:Union[int,str]=10) -> None:
        """This is the main class for the PyCalEva framework bundeling statistical tests, 
            metrics and plot for calibration measurement of binary classification models.

        Parameters
        ----------
        y_true : array_like
                Expected class labels given in test set. (Ground truth y)
        y_pred : array_like
                Observed probabilities predicted by a classification model.
        outsample : bool
                Set to 'False' for internal evaluation or set to 'True'
                for external evaluation.
        n_groups: int or str (optional, default=10)
                Number of groups to use for grouping probabilities.
                Set to 'auto' to use sturges function for estimation of optimal group size [9].

        Raises
        ------
            ValueError: If the given data (y_true,y_pred) or the given number of groups is invalid

        Examples
        --------
        >>> from pycaleva import CalibrationEvaluator
        >>> ce = CalibrationEvaluator(y_test, pred_prob, outsample=True, n_groups='auto')

        References
        ----------
        ..  [9] Sturges, H. A. (1926). The choice of a class interval. 
            Journal of the american statistical association, 21(153), 65-66.

        """
        
        # Check parameters
        self.__check_parameters(np.array(y_true), np.array(y_pred), outsample)

        self.__y = np.array(y_true) # True class labels
        self.__p = np.array(y_pred) # Predicted class probabilities

        self.__n = len(y_true)      # Sample size
        self.__ngroups = None       # Group size

        # Set if external testset or internal trainingsset is used 
        if outsample:
            self.__devel = DEVEL.EXTERNAL
        else:
            self.__devel = DEVEL.INTERNAL
        
        # Define calibration metrics
        self.__auroc = roc_auc_score(self.__y, self.__p)    # Area under the receiver operating curve
        self.__brier = brier(self.__y, self.__p,)           # Brier score
        self.__ace = None                                   # Adative calibration error
        self.__mce = None                                   # Maximum calibration error
        self.__awlc = None                                  # Area within lowess curve

        # Group data according to predicted probabilities --> will also set contengency table for groups
        self.__data = None
        self.__ct = None
        self.group_data(n_groups) # --> This method will update all groupbased metrics as well

    # PROPERTIES
    #---------------------------------------------------------------------------------------------
    @property
    def contingency_table(self):
        """Get the contingency table for grouped observed and expected class membership probabilities.
        
        Returns
        -------
            contingency_table :  DataFrame
        """
        return self.__ct

    @property
    def auroc(self):
        """Get the area under the receiver operating characteristic

        Returns
        -------
            auroc :  float
        """
        return self.__auroc

    @property
    def brier(self):
        """Get the brier score for the current y_true and y_pred of class instance.

        Returns
        -------
            brier_score :  float
        """
        return self.__brier

    @property
    def ace(self):
        """Get the adaptive calibration error based on grouped data.

        Returns
        -------
            adaptive calibration error : float
        """
        return self.__ace

    @property
    def mce(self):
        """Get the maximum calibration error based on grouped data.

        Returns
        -------
            maximum calibration error : float
        """
        return self.__mce

    @property
    def awlc(self):
        """Get the area between the nonparametric curve estimated by lowess and
            the theoritcally perfect calibration given by the calibration plot bisector.
        
        Returns
        -------
            Area within lowess curve : float
        """
        return self.__awlc

    @property
    def outsample(self):
        """Get information if outsample is set. External validation if set to 'True'.
        
        Returns
        -------
            Outsample status : bool
        """
        return self.__devel


    # PRIVATE METHODS
    # --------------------------------------------------------------------------------------------
    
    # Check if parameters are valid
    def __check_parameters(self, y, p, outsample) -> bool:
        if (len(y) != len(p)):
            raise ValueError("Observations y_true and Predictions y_pred differ in size!")
        if not ( ((y==0) | (y==1)).all() ):
            raise ValueError("Invalid class labels! y_train must be dichotomous containing only values 0 or 1")
        if ( (p < 0.0 ).any() or (p > 1.0).any() ):
            raise ValueError("Predicted probabilities y_pred must be in range [0.0 1.0]!")
        if (abs( p.sum() - y.sum() ) < 1e-04 ) and outsample == True:
            warnings.warn(Warning("Please set parameter outsample to 'false' if the evaluated model was fit on this dataset!"), "UserWarning")
        if ( y.sum() <= 1 ) or ( y.sum() >= (len(y) - 1) ):
            raise ValueError("The number of events/non events in observations can not be less than 1.")

        return True


    def __calc_ace(self):
        return np.abs((self.__ct.mean_predicted - self.__ct.mean_observed)).sum() / self.__ngroups

    def __calc_mce(self):
        return np.abs((self.__ct.mean_predicted - self.__ct.mean_observed)).max()

    def __init_contingency_table(self) -> pd.DataFrame:
        """Initialize the contingency table using data

        Returns:
            contingency_table : DataFrame:
        """
        data = self.__data
        total = data['class'].groupby(data.dcl, observed=False).count()         # Total observations per group
        mean_predicted = data['prob'].groupby(data.dcl, observed=False).mean()  # Mean predicted probability per group
        mean_observed = data['class'].groupby(data.dcl, observed=False).mean()  # Mean observed probability per group
        observed = data['class'].groupby(data.dcl, observed=False).sum()        # Number of observed class 1 events
        predicted = data['prob'].groupby(data.dcl, observed=False).sum()        # Number of predicted class 1 events

        c_table = pd.DataFrame({"total":total, "mean_predicted":mean_predicted, "mean_observed":mean_observed, \
                                "observed_0":total-observed, "predicted_0":total-predicted, 
                                "observed_1":observed, "predicted_1":predicted})
        c_table.index.rename('Interval', inplace=True) #Rename index column
        return c_table


    def __highlight_high_diff(self,row:pd.Series):
        """Highlight contingency table cells with high difference in observed and expected values
        """
        props = [f'color: black']*len(row)

        if ( abs(row.predicted_1 - row.observed_1) > (HIGHLIGHT_DEVIATION_LIMIT * row.total) ):
            props[-1] = f'color: red'

        return props


    def __warn_expected_low(self):
        """Print warning message if expected frequencies are low.
        """
        if (self.__ct.predicted_1 < CHI_SQUARE_VIOLATION_LIMIT).any():
           print(f'Warning! Some expected frequencies are smaller then {CHI_SQUARE_VIOLATION_LIMIT}. ' +
                    'Possible violoation of chi²-distribution.')


    def __show_contingency_table(self, phi=None):
        """Display the contingency table using IPython.
        """
        ct_out = self.__ct.drop(['observed_0', 'predicted_0'], axis=1).copy()

        # Add phi correction factor if values are given
        if not phi is None:
            ct_out.insert(3, "phi", phi)

        ct_out.reset_index(inplace=True)
        display(ct_out.style.apply(self.__highlight_high_diff, axis = 1))


    def __update_groupbased_metrics(self):
        """Update all metrics of class instance that are based on grouping
        """
        self.__ace = self.__calc_ace()                      # Update Adative Calibration Error
        self.__mce = self.__calc_mce()                      # Update Maximum Calibration Error

        self.__nonparametric_fit()                          # Calculate nonparametric fit and update Area Within Lowess Curve


    def __nonparametric_fit(self, update_awlc=True):
        # Nonparametric curve based on y and p using lowess
        x_nonparametric = np.arange(0,1,0.005)
        y_nonparametric = lowess(self.__y, self.__p, it=0, xvals=x_nonparametric)

        if update_awlc:
            diff = np.abs(x_nonparametric - y_nonparametric)
            self.__awlc = integrate.trapezoid(diff, x_nonparametric) # Area within loss curve
            
        return (x_nonparametric, y_nonparametric)


    def __metrics_to_string(self):
        """Returns all metrics as formatted table.

        Returns
        -------
            all_metrics: str
        """
        metrics = {"AUROC":self.__auroc, "Brier":self.__brier, "ACE":self.__ace, "MCE":self.__mce, "AWLC":self.__awlc }

        lines = ['{:<10s}{:>8d}'.format("n",self.__n)]
        for k, v in metrics.items():
            lines.append('{:<10s}{:>8.3f}'.format(k,v))
        
        textstr = '\n'.join(lines)
        return textstr


    # PUBLIC METHODS
    # --------------------------------------------------------------------------------------------
    
    # UTILITY: Return all metrics
    def metrics(self):
        """Get all available calibration metrics as combined result tuple.

        Returns
        -------
        auroc   : float
                    Area under the receiver operating characteristic.
        brier   : float
                    The scaled brier score.
        ace     : int
                    Adaptive calibration error.
        mce     : float
                    Maximum calibration error.
        awlc    : float
                    Area within the lowess curve
        
        Examples
        --------
        >>> from pycaleva import CalibrationEvaluator
        >>> ce = CalibrationEvaluator(y_test, pred_prob, outsample=True, n_groups='auto')
        >>> ce.metrics()
        metrics_result(auroc=0.9739811912225705, brier=0.2677083794415594, ace=0.0361775962446639, mce=0.1837227304691177, awlc=0.041443052220213474)
        """
        return metrics_result(self.__auroc, self.__brier, self.__ace, self.__mce, self.__awlc)


    # UTILITY: Group data
    def group_data(self, n_groups:Union[int,str]) -> None:
        r"""Group class labels and predicted probabilities into equal sized groupes of size n.

        Parameters
        ----------
        n_groups: int or str
                Number of groups to use for grouping probabilities.
                Set to 'auto' to use sturges function for estimation of optimal group size [9].

        Notes
        -----
        Sturges function for estimation of optimal group size:
        
        .. math::
            k=\left\lceil\log _{2} n\right\rceil+1

        Hosmer and Lemeshow recommend setting number of groups to 10 and with equally sized groups [1].
            
        Raises
        ------
            ValueError: If the given number of groups is invalid.
        
        References
        ----------
        ..  [1] Hosmer Jr, David W., Stanley Lemeshow, and Rodney X. Sturdivant.
                Applied logistic regression. Vol. 398. John Wiley & Sons, 2013.
        ..  [9] Sturges, H. A. (1926). The choice of a class interval. 
                Journal of the american statistical association, 21(153), 65-66.

        """

        # Check group size parameter and set accordingly
        if isinstance(n_groups, int) and 2 <= n_groups < self.__n:
            self.__ngroups = n_groups   # Group size
        elif isinstance(n_groups, str) and n_groups == 'auto':
            self.__ngroups = ceil(log2(self.__n)) + 1
        else:
            raise ValueError(f"'{n_groups}' is an invalid value of parameter n_groups!")

        df = pd.DataFrame(data={'class':self.__y, 'prob':self.__p})
        
        # Sort Values according to their probability
        df = df.sort_values('prob')
        df.reset_index(inplace=True)
        
        # Group data using deciles of risks
        try:
            df['dcl'] = pd.qcut(df['prob'], self.__ngroups)
        except ValueError:
            # Most likely low variance in probabilities results in non unique bin edges
            try:
                # FIX -> Put some probabilities into the same group
                df['dcl'] = pd.qcut(df['prob'].rank(method='first'), self.__ngroups)

                # Get propper interval labels after applied fix
                new_categories = {}
                df['rank'] = df['prob'].rank(method='first')
                #for bin in set(df.dcl):
                    #if bin.right > len(df.prob)-1:
                        #new_categories[bin] = pd.Interval(left=round(df.prob[int(bin.left)],3), right=round(df.prob.iloc[-1],3))
                    #else:
                        #new_categories[bin] = pd.Interval(left=round(df.prob[int(bin.left)],3), right=round(df.prob[int(bin.right)],3))
                    
                #df['dcl'] = df['dcl'].cat.rename_categories(new_categories)
            except ValueError:
                raise Exception("Could not create groups. Maybe try with a lower number of groups or set n_groups to 'auto'.")
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            raise

        self.__data = df
        self.__ct = self.__init_contingency_table()

        self.__update_groupbased_metrics()


    # UTILITY: Merge Groups
    def merge_groups(self, min_count:int=CHI_SQUARE_VIOLATION_LIMIT) -> None:
        """Merge groups in contingency table to have count of expected and observed class events >= min_count.

        Parameters
        ----------
        min_count : int (optional, default=1)

        Notes
        -----
        Hosmer and Lemeshow mention the possibility to merge groups at low samplesize to have higher expected and observed class event counts [1].
        This should guarantee that the requirements for chi-square goodness-of-fit tests are fullfilled.
        Be aware that the power of tests will be lower after merge!

        References
        ----------
        ..  [1] Hosmer Jr, David W., Stanley Lemeshow, and Rodney X. Sturdivant.
                Applied logistic regression. Vol. 398. John Wiley & Sons, 2013.
        """

        i = 0
        merged_rows = self.__ct.iloc[0:i].sum(axis=0, numeric_only=True)

        # Merge groups as long expected and observed count is below min_count
        while (i < self.__ngroups and (merged_rows["observed_1"] < min_count or merged_rows["predicted_1"] < min_count) ):
            merged_rows = self.__ct.iloc[0:i].sum(axis=0, numeric_only=True)
            i += 1

        # Reset index of contingency table and add merged row
        idx = pd.Interval(self.__ct.index[0].left, self.__ct.index[i-1].right)
        self.__ct.loc[idx] = merged_rows

        self.__ct = self.__ct[i:]
        self.__ct.sort_index(axis=0, inplace=True)

        # Update number of groups
        self.__ngroups = len(self.__ct)

        # Update bins in data
        self.__data['dcl'] = pd.cut(self.__data['prob'], self.__ct.index)

        # Update metrics
        self.__update_groupbased_metrics()


    # STATISTICAL TEST: Hosmer Lemeshow Test
    def hosmerlemeshow(self, verbose = True) -> hltest_result:
        r""" Perform the Hosmer-Lemeshow goodness of fit test on the data of class instance.
            The Hosmer-Lemeshow test checks the null hypothesis that the number of 
            given observed events match the number of expected events using given 
            probabilistic class predictions and dividing those into deciles of risks.
            
            Parameters
            ----------
            verbose : bool (optional, default=True)
                Whether or not to show test results and contingency table the teststatistic
                relies on.
            
            Returns
            -------
            C       : float
                        The Hosmer-Lemeshow test statistic.
            p-value : float
                        The p-value of the test.
            dof     : int
                        Degrees of freedom
            
            See Also
            --------
            CalibrationEvaluator.pigeonheyse
            CalibrationEvaluator.z_test
            scipy.stats.chisquare

            Notes
            -----
            A low value for C and high p-value (>0.05) indicate a well calibrated model.
            The power of this test is highly dependent on the sample size. Also the 
            teststatistic lacks fit to chi-squared distribution in some situations [3]. 
            In order to decide on model fit it is recommended to check it's discrematory
            power as well using metrics like AUROC, precision, recall. Furthermore a
            calibration plot (or reliability plot) can help to identify regions of the
            model underestimate or overestimate the true class membership probabilities.
            
            Hosmer and Lemeshow estimated the degrees of freedom for the teststatistic
            performing extensive simulations. According to their results the degrees of 
            freedom are k-2 where k is the number of subroups the data is divided into. 
            In the case of external evaluation the degrees of freedom is the same as k [1]. 
            
            Teststatistc:

                .. math:: 
                    E_{k 1}=\sum_{i=1}^{n_{k}} \hat{p}_{i 1}

                .. math:: 
                    O_{k 1}=\sum_{i=1}^{n_{k}} y_{i 1}

                .. math:: 
                    \hat{C}=\sum_{k=1}^{G} \frac{\left(O_{k 1}-E_{k 1}\right)^{2}}{E_{k 1}} + \frac{\left(O_{k 0}-E_{k 0}\right)^{2}}{E_{k 0}}

            References
            ----------
            ..  [1] Hosmer Jr, David W., Stanley Lemeshow, and Rodney X. Sturdivant. 
                Applied logistic regression. Vol. 398. John Wiley & Sons, 2013.
            ..  [10] "Hosmer-Lemeshow test", https://en.wikipedia.org/wiki/Hosmer-Lemeshow_test
            ..  [11] Pigeon, Joseph G., and Joseph F. Heyse. "A cautionary note about assessing 
                the fit of logistic regression models." (1999): 847-853.

            Examples
            --------
            >>> from pycaleva import CalibrationEvaluator
            >>> ce = CalibrationEvaluator(y_test, pred_prob, outsample=True, n_groups='auto')
            >>> ce.hosmerlemeshow()
            hltest_result(statistic=4.982635477424991, pvalue=0.8358193332183672, dof=9)
        """

        # Calculate Hosmer Lemeshow Teststatistic based on contengency table
        C_ = ( (self.__ct.observed_1 - self.__ct.predicted_1)**2 / \
                (self.__ct.total*self.__ct.mean_predicted*(1-self.__ct.mean_predicted)) ).sum()
        
        # DoF Internal = Number Subgroups - Parameters of Logistic Regression [1]
        # DoF External = Number Subgroups [1]
        if self.__devel == DEVEL.INTERNAL:
            dof = self.__ngroups-2
        else:
            dof = self.__ngroups
        
        # Calculate pvalue
        pval = 1 - chi2.cdf(C_, dof)
        
        # Show the contingency table
        if verbose:
            self.__show_contingency_table()
        
            # Warn user if expected frequencies are < 5
            self.__warn_expected_low()

            if (pval < 0.001):
                print(f'C({dof}): {C_:.2f} p-value: < 0.001')
            else:
                print(f'C({dof}): {C_:.2f} p-value: {pval:.3f}')
        
        return hltest_result(C_, pval, dof)


    # STATISTICAL TEST: Pigeon Heyse Test
    def pigeonheyse(self, verbose = True) -> phtest_result:
        r"""Perform the Pigeon-Heyse goodness of fit test.
        The Pigeon-Heyse test checks the null hypothesis that number of given observed 
        events match the number of expected events over divided subgroups.
        Unlike the Hosmer-Lemeshow test this test allows the use of different
        grouping strategies and is more robust against variance within subgroups.
        
        Parameters
        ----------
        verbose : bool (optional, default=True)
                Whether or not to show test results and contingency table the teststatistic
                relies on.

        Returns
        -------
        J : float
            The Pigeon-Heyse test statistic J².
        p : float
            The p-value of the test.
        dof : int
                Degrees of freedom
        
        See Also
        --------
        CalibrationEvaluator.hosmerlemeshow
        CalibrationEvaluator.z_test
        scipy.stats.chisquare

        Notes
        -----
        This is an implemenation of the test proposed by Pigeon and Heyse [2].
        A low value for J² and high p-value (>0.05) indicate a well calibrated model.
        Other then the Hosmer-Lemeshow test an adjustment factor is added to
        the calculation of the teststatistic, making the use of different 
        grouping strategies possible as well.
        
        The power of this test is highly dependent on the sample size.
        In order to decide on model fit it is recommended to check it's discrematory
        power as well using metrics like AUROC, precision, recall. Furthermore a
        calibration plot (or reliability plot) can help to identify regions of the
        model underestimate or overestimate the true class membership probabilities.
        
        Teststatistc:

            .. math:: 
                \phi_{k}=\frac{\sum_{i=1}^{n_{k}} \hat{p}_{i 1}\left(1-\hat{p}_{i 1}\right)}{n_{k} \bar{p}_{k 1}\left(1-\bar{p}_{k 1}\right)}

            .. math:: 
                {J}^{2}=\sum_{k=1}^{G} \frac{\left(O_{k 1}-E_{k 1}\right)^{2}}{\phi_{k} E_{k 1}} + \frac{\left(O_{k 0}-E_{k 0}\right)^{2}}{\phi_{k} E_{k 0}}

        References
        ----------
        ..  [1] Hosmer Jr, David W., Stanley Lemeshow, and Rodney X. Sturdivant. 
            Applied logistic regression. Vol. 398. John Wiley & Sons, 2013.
        ..  [2] Pigeon, Joseph G., and Joseph F. Heyse. "An improved goodness of 
            fit statistic for probability prediction models."
            Biometrical Journal: Journal of Mathematical Methods in Biosciences 
            41.1 (1999): 71-82.
        ..  [11] Pigeon, Joseph G., and Joseph F. Heyse. "A cautionary note about assessing 
            the fit of logistic regression models." (1999): 847-853.
        
        Examples
        --------
        >>> from pycaleva import CalibrationEvaluator
        >>> ce = CalibrationEvaluator(y_test, pred_prob, outsample=True, n_groups='auto')
        >>> ce.pigeonheyse()
        phtest_result(statistic=5.269600396341568, pvalue=0.8102017228852412, dof=9)
        """
        
        # Factor phi to adjust X² statistic
        phi = ( self.__data['prob'].groupby(self.__data.dcl, observed=False).apply(lambda x: (x *(1-x)).sum()) ) /  \
            ( self.__ct.total * self.__ct.mean_predicted * (1 - self.__ct.mean_predicted) )

        # Teststatistic
        J_square = ( (self.__ct.observed_1 - self.__ct.predicted_1)**2 / \
            (phi*self.__ct.total*self.__ct.mean_predicted*(1-self.__ct.mean_predicted)) ).sum()
        

        # DoF Internal = Number Subgroups - 1 [2]
        # DoF External = Number Subgroups [1] 
        if self.__devel == DEVEL.INTERNAL:
            dof = self.__ngroups - 1
        else:
            dof = self.__ngroups


        pval = 1 - chi2.cdf(J_square, dof)  # Calculate pvalue
        
        if verbose:
            # Show the contingency table
            self.__show_contingency_table(phi)

            # Warn user if expected frequencies are < 5
            self.__warn_expected_low()

            if (pval < 0.001):
                print(f'J²({dof}): {J_square:.2f} p-value: < 0.001')
            else:
                print(f'J²({dof}): {J_square:.2f} p-value: {pval:.3f}')

        return phtest_result(J_square, pval, dof)


    # STATISTICAL TEST: Spiegelhalter z-test
    def z_test(self) -> ztest_result:
        r"""Perform the Spieglhalter's z-test for calibration.
        
        Returns
        -------
        statistic : float
            The Spiegelhalter z-test statistic.
        p : float
            The p-value of the test.
        

        See Also
        --------
        CalibrationEvaluator.hosmerlemeshow
        CalibrationEvaluator.pigeonheyse


        Notes
        -----
        This calibration test is performed in the manner of a z-test. 
        The nullypothesis is that the estimated probabilities are equal to the true class probabilities.
        The test statistic under the nullypothesis can be approximated by a normal distribution. 
        A low value for Z and high p-value (>0.05) indicate a well calibrated model.
        Other than Hosmer Lemeshow Test or Pigeon Heyse Test, this test is not based on grouping strategies.

        Teststatistc:

            .. math::
                Z=\frac{\sum_{i=1}^{n}\left(y_{i}-\hat{p}_{i}\right)\left(1-2 \hat{p}_{i}\right)}{\sqrt{\sum_{i=1}^{n}\left(1-2 \hat{p}_{i}\right)^{2} \hat{p}_{i}\left(1-\hat{p}_{i}\right)}}
            
        
        References
        ----------
        ..  [1] Spiegelhalter, D. J. (1986). Probabilistic prediction in patient management and clinical trials. 
            Statistics in medicine, 5(5), 421-433.
        ..  [2] Huang, Y., Li, W., Macheret, F., Gabriel, R. A., & Ohno-Machado, L. (2020). 
            A tutorial on calibration measurements and calibration models for clinical prediction models. 
            Journal of the American Medical Informatics Association, 27(4), 621-633.
        

        Examples
        --------
        >>> from pycaleva import CalibrationEvaluator
        >>> ce = CalibrationEvaluator(y_test, pred_prob, outsample=True, n_groups='auto')
        >>> ce.z_test()
        ztest_result(statistic=-0.21590257919669287, pvalue=0.829063686607032)

        """

        num = ( (self.__y - self.__p) * ( 1 - 2 * self.__p )  ).sum()
        denom = sqrt( ((1 - 2 * self.__p)**2 * self.__p * ( 1 - self.__p)).sum() )

        z = num / denom
        pval = 2 * norm.cdf(-abs(z))

        return ztest_result(z, pval)


    # STATISTICAL TEST / PLOT : Calibration Belt
    def calbelt(self, plot:bool=False, subset = None, confLevels=[0.8, 0.95], alpha=0.95) -> calbelt_result:
        """Calculate the calibration belt and draw plot if desired.
        
        Parameters
        ----------
        plot: boolean, optional
            Decide if plot for calibration belt should be shown.
            Much faster calculation if set to 'false'!
        subset: array_like
            An optional boolean vector specifying the subset of observations to be considered.
            Defaults to None.
        confLevels: list
            A numeric vector containing the confidence levels of the calibration belt.
            Defaults to [0.8,0.95].
        alpha: float
            The level of significance to use.

        Returns
        -------
        T : float
            The Calibration plot test statistic T.
        p : float
            The p-value of the test.
        fig : matplotlib.figure
            The calibration belt plot. Only returned if plot='True'
        
        See Also
        --------
        pycaleva.calbelt.CalibrationBelt
        CalibrationEvaluator.calplot
        

        Notes
        -----
        This is an implemenation of the test proposed by Nattino et al. [6]. 
        The implementation was built upon the python port of the R-Package givitiR [8] and the python implementation calibration-belt [7].
        The calibration belt estimates the true underlying calibration curve given predicted probabilities and true class labels.
        Instead of directly drawing the calibration curve a belt is drawn using confidence levels.
        A low value for the teststatistic and a high p-value (>0.05) indicate a well calibrated model.
        Other than Hosmer Lemeshow Test or Pigeon Heyse Test, this test is not based on grouping strategies.

        References
        ----------
        ..  [6] Nattino, G., Finazzi, S., & Bertolini, G. (2014). A new calibration test 
            and a reappraisal of the calibration belt for the assessment of prediction models 
            based on dichotomous outcomes. Statistics in medicine, 33(14), 2390-2407.

        ..  [7] Bulgarelli, L. (2021). calibrattion-belt: Assessment of calibration in binomial prediction models [Computer software].
            Available from https://github.com/fabiankueppers/calibration-framework

        ..  [8] Nattino, G., Finazzi, S., Bertolini, G., Rossi, C., & Carrara, G. (2017).
            givitiR: The giviti calibration test and belt (R package version 1.3) [Computer
            software]. The Comprehensive R Archive Network.
            Available from https://CRAN.R-project.org/package=givitiR
        

        Examples
        --------
        >>> from pycaleva import CalibrationEvaluator
        >>> ce = CalibrationEvaluator(y_test, pred_prob, outsample=True, n_groups='auto')
        >>> ce.calbelt(plot=False)
        calbelt_result(statistic=1.6111330037643796, pvalue=0.4468347221346196, fig=None)

        """
        
        cb = CalibrationBelt(self.__y, self.__p, self.__devel, subset=subset, confLevels=confLevels, alpha=alpha)
        
        if plot:
            return cb.plot()
        else:
            return cb.stats()


    def calibration_plot(self):
        """Generate the calibration plot for the given predicted probabilities and true class labels of current class instance.

        Returns
        -------
            plot : matplotlib.figure

        Notes
        -----
        This calibration plot is showing the predicted class probability against the actual probability according to the true class labels 
        as a red triangle for each of the groups. An additional calibration curve is draw, estimated using the LOWESS algorithm. 
        A model is well calibrated, if the red triangles and the calibration curve are both close to the plots bisector.
        In the left corner of the plot all available metrics are listed as well. This implementation was made following the example of the R package
        rms [5].

        See Also
        --------
        CalibrationEvaluator.calbelt

        References
        ----------
        ..  [5] Jr, F. E. H. (2021). rms: Regression modeling strategies (R package version
            6.2-0) [Computer software]. The Comprehensive R Archive Network.
            Available from https://CRAN.R-project.org/package=rms

        Examples
        --------
        >>> from pycaleva import CalibrationEvaluator
        >>> ce = CalibrationEvaluator(y_test, pred_prob, outsample=True, n_groups='auto')
        >>> ce.calibration_plot()

        """
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.set_ylim(0.0, 1.05)

        # Draw a calibration plot using matplotlib only
        y_grouped = self.__ct["mean_observed"]
        p_grouped = self.__ct["mean_predicted"]

        # Get nonparametric curve based on y and p using lowess
        x_nonparametric,y_nonparametric = self.__nonparametric_fit(update_awlc=False)

        # Add calibration line for model
        plt.scatter(p_grouped,y_grouped, marker="^", facecolors='none', edgecolors='r', label=f'Grouped observations g={self.__ngroups}')

        # Add line for nonparametric fit using lowess
        ax1.plot(x_nonparametric, y_nonparametric, label="Nonparametric")

        # Add line for perfect calibration
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        # Add stem plot for distribution of observations
        h, e = np.histogram(self.__p, bins=50, density=False)
        bin_center_points = 0.5 * (e[1:] + e[:-1])
        h = h.astype('float')
        h /= h.sum()                   # normalize stem height to sum to 1
        ax2 = ax1.twinx()
        ax2.yaxis.set_visible(False)
        ax2.set_ylim([-0.01, 1.05])    # y-domain for the stem plot goes from 0 to 1, with slight offsets for visibility reasons
        ax2.stem(bin_center_points, h, linefmt="grey", markerfmt=" ", basefmt=" ")

        ax1.set_xlabel('Predicted Probability')
        ax1.set_ylabel('Actual Probability')
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax1.text(0.02, 0.78, self.__metrics_to_string(), fontsize=10, family='monospace', bbox=props)

        ax1.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5), fancybox=True, framealpha=0.5)

        return fig
