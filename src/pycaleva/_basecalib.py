"""
A framework for calibration assessment of binary classification models 
written in Python.

References
----------
[Hosmer Jr, David W., Stanley Lemeshow, and Rodney X. Sturdivant. 
 Applied logistic regression. Vol. 398. John Wiley & Sons, 2013.]

[Pigeon, Joseph G., and Joseph F. Heyse. "An improved goodness of 
fit statistic for probability prediction models." Biometrical Journal: 
Journal of Mathematical Methods in Biosciences 41.1 (1999): 71-82.]

[Nattino, G., Finazzi, S., & Bertolini, G. (2014). A new calibration 
test and a reappraisal of the calibration belt for the assessment of 
prediction models based on dichotomous outcomes. Statistics in medicine, 
33(14), 2390-2407.]

[Sturges, H. A. (1926). The choice of a class interval. 
Journal of the american statistical association, 21(153), 65-66.]


TODO
----------
* Improve Calibration belt performance (boundary calculation)
* Get all metrics as namedtuple
* Make possible for list input as well (y,p)
* Exception handling --> Check parameters --> Take over from CalibrationBelt eventually, Check for NaN Values
* Bin edges must be unique when few values and low variance
* Improve Docstrings (Examples are wrong due to changes, Some docstrings missing)
* Add option to use grouping vector (clustering before calibration test)
* Low number of groups warning ( at g<6 )
* LOWESS Curve does not fit in comparison to R package rms
"""

from argparse import ArgumentError
from enum import Flag
from math import log2, ceil, sqrt
from collections import namedtuple
from typing import Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from scipy.stats import chi2, norm
from scipy import integrate
from statsmodels.nonparametric.smoothers_lowess import lowess
from IPython.display import display

from .calbelt import CalibrationBelt, calbelt_result
from .metrics import brier, auroc, brier_skill_score


__all__ = ['CalibrationEvaluator']


# SETS THE LIMIT FOR THE FREQUENCY IN CONTINGENCY TABLES
CHI_SQUARE_VIOLATION_LIMIT = 1


# DEFINE RETURN TYPES
hltest_result = namedtuple('hltest_result', ['statistic', 'pvalue', 'dof'])
phtest_result = namedtuple('phtest_result', ['statistic', 'pvalue', 'dof'])
ztest_result = namedtuple('ztest_result', ['statistic', 'pvalue'])



class DEVEL(Flag):
    INTERNAL = False
    EXTERNAL = True



class _BaseCalibrationEvaluator:

    def __init__(self, y_true:np.ndarray, y_pred:np.ndarray, outsample:bool, n_groups:Union[int,str]=10) -> None:
        """This is a framework for calibration measurement of binary classification models.

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
                Set to 'auto' to use sturges function for estimation of optimal group size [1].

        References
        ----------
        .. [1] [Sturges, H. A. (1926). The choice of a class interval. 
            Journal of the american statistical association, 21(153), 65-66.]

        """
        # Sets the output precision for floats
        pd.options.display.precision = 3

        # Check if classlabels are dichotomous
        if not set(y_true) == set([0,1]):
            raise ValueError("Invalid class labels! y_train must be dichotomous containing only values 0 or 1")

        # Check if all probabilities betwenn 0.0 and 1.0
        if not all(x >= 0.0 and x <= 1.0 for x in y_pred):
            raise ValueError("Predicted probabilities must be in range [0.0 1.0]!")


        self.__y = y_true           # True class labels
        self.__p = y_pred           # Predicted class probabilities
        self.__n = len(y_true)      # Sample size
        self.__ngroups = None       # Group size

        # Set if external testset or internal trainingsset is used 
        if outsample:
            self.__devel = DEVEL.EXTERNAL
        else:
            self.__devel = DEVEL.INTERNAL
        
        # Define calibration metrics
        self.__auroc = auroc(self.__y, self.__p)            # Area under the receiver operating curve
        self.__brier = brier(self.__y, self.__p, True)      # Brier score scaled to [0.0 - 1.0]
        self.__ace = None                                   # Adative calibration error
        self.__mce = None                                   # Maximum calibration error
        self.__awlc = None                                  # Area within lowess curve

        # Group data according to predicted probabilities and get contengency table for groups
        self.__data = None
        self.__ct = None
        self.group_data(n_groups)

        
    def __calc_ace(self):
        return np.abs((self.__ct.mean_predicted - self.__ct.mean_observed)).sum() / self.__ngroups

    def __calc_mce(self):
        return np.abs((self.__ct.mean_predicted - self.__ct.mean_observed)).max()

    @property
    def contingency_table(self):
        return self.__ct

    @property
    def brier(self):
        return self.__brier

    @property
    def ace(self):
        return self.__ace

    @property
    def mce(self):
        return self.__mce

    @property
    def outsample(self):
        return self.__devel

    def group_data(self, n_groups:Union[int,str]) -> None:

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
        
        # Group data using deciles of risks
        df['dcl'] = pd.qcut(df['prob'], self.__ngroups)

        self.__data = df
        self.__ct = self.__init_contingency_table()

        self.__update_groupbased_metrics()



    def merge_groups(self, min_count:int=CHI_SQUARE_VIOLATION_LIMIT) -> None:
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


    def __init_contingency_table(self) -> pd.DataFrame:
        data = self.__data
        total = data['class'].groupby(data.dcl).count()         # Total observations per group
        mean_predicted = data['prob'].groupby(data.dcl).mean()  # Mean predicted probability per group
        mean_observed = data['class'].groupby(data.dcl).mean()  # Mean observed probability per group
        observed = data['class'].groupby(data.dcl).sum()        # Number of observed class 1 events
        predicted = data['prob'].groupby(data.dcl).sum()        # Number of predicted class 1 events

        c_table = pd.DataFrame({"total":total, "mean_predicted":mean_predicted, "mean_observed":mean_observed, \
                                "observed_1":observed, "predicted_1":predicted})
        c_table.index.rename('Interval', inplace=True) #Rename index column
        return c_table

    def __highlight_expected_low(self,row:pd.Series):
        props = [f'color: black']*len(row)

        if row.predicted_1 < CHI_SQUARE_VIOLATION_LIMIT:
            props[-1] = f'color: red'

        return props

    def __warn_expected_low(self):
        if (self.__ct.predicted_1 < CHI_SQUARE_VIOLATION_LIMIT).any():
            print(f'Warning! Some expected frequencies are smaller then {CHI_SQUARE_VIOLATION_LIMIT}. ' +
                    'Possible violoation of chi²-distribution.')

    def __show_contingency_table(self, phi=None):
        ct_out = self.__ct.copy()

        # Add phi correction factor if values are given
        if not phi is None:
            ct_out.insert(3, "phi", phi)

        ct_out.reset_index(inplace=True)
        display(ct_out.style.apply(self.__highlight_expected_low, axis = 1))


    def hosmerlemeshow(self, verbose = True) -> tuple[float,float]:
        """ Calculate a Hosmer-Lemeshow goodness of fit test.
            The Hosmer-Lemeshow test checks the null hypothesis that the number of 
            given observed events match the number of expected events using given 
            probabilistic class predictions and dividing those into deciles of risks.
            
            Parameters
            ----------
            show_ct : bool (optional, default=True)
                Whether or not to show test results and contingency table the teststatistic
                relies on.
            
            Returns
            -------
            c : float
                The Hosmer-Lemeshow test statistic.
            p : float
                The p-value of the test.
            dof : int
                Degrees of freedom
            
            See Also
            --------
            CalibrationEvaluator.pigeonheyse
            scipy.stats.chisquare

            Notes
            -----
            The power of this test is highly dependent on the sample size. Also the 
            teststatistic lacks fit to chi-squared distribution in some situations [3]. 
            In order to decide on model fit it is recommended to check it's discrematory
            power as well using metrics like AUROC, precision, recall. Furthermore a
            calibration plot (or reliability plot) can help to identify regions of the
            model underestimate or overestimate the true class membership probabilities.
            
            Hosmer and Lemeshow estimated the degrees of freedom [1] for the teststatistic
            performing extensive simulations. According to their results the degrees of 
            freedom are k-2 where k is the number of subroups the data is divided into. 
            
            References
            ----------
            .. [1] Hosmer Jr, David W., Stanley Lemeshow, and Rodney X. Sturdivant. 
                Applied logistic regression. Vol. 398. John Wiley & Sons, 2013.
            .. [2] "Hosmer-Lemeshow test", https://en.wikipedia.org/wiki/Hosmer-Lemeshow_test
            .. [3] Pigeon, Joseph G., and Joseph F. Heyse. "A cautionary note about assessing 
                the fit of logistic regression models." (1999): 847-853.
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


    def pigeonheyse(self, verbose = True) -> tuple[float,float]:
        """Calculate Pigeon-Heyse goodness of fit test.
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
        scipy.stats.chisquare

        Notes
        -----
        This is an implemenation of the test proposed by Pigeon and Heyse [2].
        Other then the Hosmer-Lemeshow test an adjustment factor is added to
        the calculation of the teststatistic, making the use of different 
        grouping strategies possible as well.
        TODO: DESCRIBE GROUPING STRATEGIES!
        
        The power of this test is highly dependent on the sample size.
        In order to decide on model fit it is recommended to check it's discrematory
        power as well using metrics like AUROC, precision, recall. Furthermore a
        calibration plot (or reliability plot) can help to identify regions of the
        model underestimate or overestimate the true class membership probabilities.
        
        References
        ----------
        .. [1] Hosmer Jr, David W., Stanley Lemeshow, and Rodney X. Sturdivant. 
            Applied logistic regression. Vol. 398. John Wiley & Sons, 2013.
        .. [2] Pigeon, Joseph G., and Joseph F. Heyse. "An improved goodness of 
            fit statistic for probability prediction models."
            Biometrical Journal: Journal of Mathematical Methods in Biosciences 
            41.1 (1999): 71-82.
        .. [3] Pigeon, Joseph G., and Joseph F. Heyse. "A cautionary note about assessing 
            the fit of logistic regression models." (1999): 847-853.
        

        Examples
        --------
        With just n_groups given, the test will be performed using 10 groups
        formed using the sorted class membership probabilities.
        >>> from caltest import pigeonheyse
        >>> pigeonheyse([0,1,0,1,1,0,1,0,0,0],[0.2,0.8,0.1,0.9,0.8,0.6,0.7,0.4,0.5,0.55])
        (5.7896825396825395, 0.7607692685172225)
        
        Setting groups to another value is possible but is not recommended, because of 
        undefined behaviour and unprecise estimation of degrees of freedom.
        >>> from caltest import pigeonheyse
        >>> pigeonheyse([0,1,0,1,1,0,1,0,0,0],[0.2,0.8,0.1,0.9,0.8,0.6,0.7,0.4,0.5,0.55], groups=5)
        (5.761521150308212, 0.21767994956402437)
        
        It is possible to decide on the grouping strategy to use.
        TODO: ADD EXAMPLES + DESCRIPTION
        """
        
        # Factor phi to adjust X² statistic
        phi = ( self.__data['prob'].groupby(self.__data.dcl).apply(lambda x: (x *(1-x)).sum()) ) /  \
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


    def z_test(self):
        """Calculate the Spieglhalter's z-test for calibration.
        
        Parameters
        ----------

        Returns
        -------
        statistic : float
            The Spiegelhalter z-test statistic.
        p : float
            The p-value of the test.
        
        See Also
        --------


        Notes
        -----

        
        References
        ----------
        .. [1] Spiegelhalter, D. J. (1986). Probabilistic prediction in patient management and clinical trials. 
            Statistics in medicine, 5(5), 421-433.
        .. [2] Huang, Y., Li, W., Macheret, F., Gabriel, R. A., & Ohno-Machado, L. (2020). 
            A tutorial on calibration measurements and calibration models for clinical prediction models. 
            Journal of the American Medical Informatics Association, 27(4), 621-633.
        

        Examples
        --------

        """

        num = ( (self.__y - self.__p) * ( 1 - 2 * self.__p )  ).sum()
        denom = sqrt( ((1 - 2 * self.__p)**2 * self.__p * ( 1 - self.__p)).sum() )

        z = num / denom
        pval = 2 * norm.cdf(-abs(z))

        return ztest_result(z, pval)


    def calbelt(self, plot:bool=False, confLevels:list=[0.8,0.95]) -> tuple[float,float]:
        """Calculate the calibration belt.
        
        Parameters
        ----------
        plot: boolean, optional
            Decide if plot for calibration belt should
            be shown.
        confLevels: list, optional
            Set the confidence intervalls for belt.
            Defaults to [0.8,0.95].

        Returns
        -------
        T : float
            The Calibration plot test statistic T.
        p : float
            The p-value of the test.
        
        See Also
        --------
        CalibrationEvaluator.calplot

        Notes
        -----
        This is an implemenation of the test proposed by Nattine et. al. [1]
        
        References
        ----------
        .. [1] Nattino, G., Finazzi, S., & Bertolini, G. (2014). A new calibration test 
        and a reappraisal of the calibration belt for the assessment of prediction models 
        based on dichotomous outcomes. Statistics in medicine, 33(14), 2390-2407.
        .. [2] https://github.com/fabiankueppers/calibration-framework
        .. [3] https://cran.r-project.org/web/packages/givitiR/vignettes/givitiR.html
        
        Examples
        --------
    	TODO: ADD EXAMPLES
        """
        
        cb = CalibrationBelt(self.__y, self.__p, self.__devel, confLevels=confLevels)
        
        if plot:
            return cb.plot()
        else:
            return cb.stats()


    def __update_groupbased_metrics(self):
        self.__ace = self.__calc_ace()                      # Update Adative calibration error
        self.__mce = self.__calc_mce()                      # Update Maximum calibration error


    def __metrics_to_string(self):
        metrics = {"AUROC":self.__auroc, r"$Brier_{scaled}$  ":self.__brier, "ACE":self.__ace, "MCE":self.__mce, "AWLC":self.__awlc }

        lines = ['{:<10s}{:>8d}'.format("n",self.__n)]
        for k, v in metrics.items():
            lines.append('{:<10s}{:>8.3f}'.format(k,v))
        
        textstr = '\n'.join(lines)
        return textstr


    def calibration_plot(self, verbose=True):
        fig, ax1 = plt.subplots(figsize=(10,6))

        # Draw a calibration plot using matplotlib only
        y_grouped = self.__ct["mean_observed"]
        p_grouped = self.__ct["mean_predicted"]

        # Nonparametric curve based on y and p using lowess
        x_nonparametric = np.arange(0,1,0.005)
        y_nonparametric = lowess(self.__y, self.__p, it=0, xvals=x_nonparametric)

        diff = np.abs(x_nonparametric - y_nonparametric)
        self.__awlc = integrate.trapezoid(diff, y_nonparametric) # Area within loss curve

        # Add calibration line for model
        plt.scatter(p_grouped,y_grouped, marker="^", facecolors='none', edgecolors='r', label='Grouped observations')

        # Add histogram on second axis
        h, e = np.histogram(self.__p, bins=50)
        h = h.astype('float')
        h /= h.max()                # Get relative frequencies
        ax2 = ax1.twinx()
        ax2.set_ylim(-0.1,5)        # Scale down histogram
        ax2.axis('off')             # Hide labels and ticks
        ax2.stem(e[:-1],h, linefmt="grey", markerfmt=" ", basefmt=" ")

        # Add line for nonparametric fit using lowess
        ax1.plot(x_nonparametric, y_nonparametric, label="Nonparametric")

        # Add line for perfect calibration
        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

        ax1.set_xlabel('Predicted Probability')
        ax1.set_ylabel('Actual Probability')
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.4)
        ax1.text(0.00, 0.75, self.__metrics_to_string(), fontsize=10, family='monospace', bbox=props)

        ax1.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))

        if verbose:
            plt.show()

        return fig