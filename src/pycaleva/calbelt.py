from math import sqrt, exp, pi, asin, atan, acos
import warnings

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from scipy.stats import chi2
from scipy.special import logit, xlogy, expit
from scipy.integrate import quad as integrate
from scipy.optimize import brentq, minimize, NonlinearConstraint

import statsmodels.api as sm
import statsmodels.formula.api as smf

from ._result_types import *


# DEFINE Cumulative density functions for m degrees 1 to 4
#----------------------------------------------------------

# Some scary math :(
def cdf_m(T:float, m:int, outsample:bool, alpha:float):
    """Defines the cumulative density functions for calibration belt of m degrees 1 to 4.
        Available for external or internal evaluation of models.

    Parameters
    ----------
    T: float
        The calibration belt teststatistic.
    m: int
        Degress of polynomial fit.
    outsample : bool
        Set to True for 'external' or False for 'internal' evaluation.
    alpha : float
        The level of significance to use.


    Returns:
    -------
        cdf : float
            The cdf value for calibration belt according to parameters. 
    """
    pDegInc = 1 - alpha
    k = chi2.ppf(1-pDegInc, 1)

    # EXTERNAL EVALUATION
    if (outsample):
        if (T <= (m-1) * k ):
            return 0
        else:
            if m == 1:
                # EXTERNAL m=1
                return chi2.cdf(T, 2)

            elif m == 2:
                # EXTERNAL m=2
                return (
                    ((chi2.cdf(T, df = 1) - 1 + pDegInc + 
                     (-1) * sqrt(2)/sqrt(pi) * exp(-T/2) * (sqrt(T) - 
                    sqrt(k)))/pDegInc)
                )

            elif m == 3:
                # EXTERNAL m=3
                integrand1 = ( lambda y: (chi2.cdf(T - y, df = 1) - 1 + pDegInc) * chi2.pdf(y, df = 1) )
                integrand2 = ( lambda y: (sqrt(T - y) - sqrt(k)) * 1/sqrt(y) )

                integral1 = integrate(integrand1, a = k, b = T - k)[0]
                integral2 = integrate(integrand2, a = k, b = T - k)[0]

                num = (integral1 - exp(-T/2)/(2 * pi) * 2 * integral2)
                den = pDegInc**2
                return (num/den)

            elif m == 4:
                # EXTERNAL m=4
                integrand = ( lambda r: ( (r**2 * (exp(-(r**2)/2) - exp(-T/2)) * 
                    (-pi * sqrt(k)/(2 * r) + 2 * sqrt(k)/r * 
                    asin((r**2/k - 1)**(-1/2)) - 2 * atan((1 - 
                    2 * k/r**2)**(-1/2)) + 2 * sqrt(k)/r * atan((r**2/k - 
                    2)**(-1/2)) + 2 * atan(r/sqrt(k) * sqrt(r**2/k - 
                    2)) - 2 * sqrt(k)/r * atan(sqrt(r**2/k - 
                    2)))) )
                )
                
                integral = integrate(integrand, a = sqrt(3 * k), b = sqrt(T))[0]

                return ((2/(pi * pDegInc**2))**(3/2) * integral)

            else:
                # EXTERNEAL m>4 is not defined!
                return NotImplemented

    # INTERNAL EVALUATION
    else:
        if (T <= (m-2) * k):
            return 0
        else:
            if m == 1:
                # INTERNAL m=1 --> Not defined as polynial fit starts with degree 2 for internal evaluation
                return NotImplemented

            elif m == 2:
                # INTERNAL m=2
                return chi2.cdf(T, 1)

            elif m == 3:
                # INTERNAL m=3
                integrand = ( lambda r: (r * exp(-(r**2)/2) * acos(sqrt(k)/r)) )
                
                integral = integrate(integrand, a = sqrt(k), b = sqrt(T))[0]
                return (2/(pi * pDegInc) * integral)

            elif m == 4:
                # INTERNAL m=4
                integrand = (lambda r: (r**2 * exp(-(r**2)/2) * (atan(sqrt(r**2/k * 
                                (r**2/k - 2))) - sqrt(k)/r * atan(sqrt(r**2/k - 
                                2)) - sqrt(k)/r * acos((r**2/k - 1)**(-1/2))))
                            )
                
                integral = integrate(integrand, a = sqrt(2 * k), b = sqrt(T))[0]
                return ((2/pi)**(3/2) * (pDegInc)**(-2) * integral)
            else:
                # INTERNAL m>4 is not defined!
                return NotImplemented



# MAIN Class for CalibrationBelt Test and Plotting
#-----------------------------------------------------------------

class CalibrationBelt():

    def __init__(self, y_true:np.ndarray, y_pred:np.ndarray, outsample:bool, subset = None, confLevels=[0.8, 0.95], alpha=0.95):
        """Calculate the calibration belt and draw plot if desired.
        
        Parameters
        ----------
        y_true : array_like
            Expected class labels given in test set. (Ground truth y)
        y_pred : array_like
            Observed probabilities predicted by a classification model.
        outsample : bool
            Set to 'False' for internal evaluation or set to 'True'
            for external evaluation.
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
        >>> from pycaleva import CalibrationBelt
        >>> cb = CalibrationBelt(y_test, pred_prob, outsample=True)
        >>> cb.stats()
        >>> cb.plot()
        """
        
        # Check parameters
        self.__check_parameters(y_true,y_pred,outsample,alpha)

        if not subset is None:
            # Pick a random subset from data
            idx = np.random.choice(np.arange(len(y_true)), subset, replace=False)
            self.__y = y_true[idx]
            self.__p = y_pred[idx]
        else:
            self.__y = y_true
            self.__p = y_pred

        self.__clip_probs()
        self.__n = len(self.__y)

        # Warn user at internal evaluation
        if not outsample:
            warnings.warn("Evaluation only valid for Logistic Regression Models");

        self.__outsample = outsample

        self.__confLevels = sorted(confLevels, reverse=True)
        self.__logit_e = logit(self.__p)   # Get logit form of predicted probabilities

        # Find polynomial fit using forward selection procedure
        self.__maxDeg = 4
        self.__model, self.__m =  self.__forward_select(alpha, self.__maxDeg)

        # Get teststatistic and p-value
        self.__T, self.__pval = self.__test(alpha)

        self.__boundaries = {}


    def __check_parameters(self, y, p, outsample, alpha) -> bool:
        if (len(y) != len(p)):
            raise ValueError("Observations y_true and Predictions y_pred differ in size!")
        if not ( ((y==0) | (y==1)).all() ):
            raise ValueError("Invalid class labels! y_train must be dichotomous containing only values 0 or 1")
        if ( (p < 0.0 ).any() or (p > 1.0).any() ):
            raise ValueError("Predicted probabilities y_pred must be in range [0.0 1.0]!")
        if (alpha < 0.0 or alpha > 1.0):
            raise ValueError("Alpha must be in range 0.0 - 1.0")
        if (abs( p.sum() - y.sum() ) < 1e-04 ) and outsample == True:
            warnings.warn("Please set parameter outsample to 'false' if the evaluated model was fit on this dataset!", "UserWarning")
        if ( y.sum() <= 1 ) or ( y.sum() >= (len(y) - 1) ):
            raise ValueError("The number of events/non events in observations can not be less than 1.")

        return True

    # Avoid probabilities to be exactly zero or one
    def __clip_probs(self):
        self.__p = np.clip(self.__p, 1e-10, 1-(1e-10))
    

    # Find polynomial fit using forward selection process
    def __forward_select(self, alpha, maxDeg):
        family = sm.families.Binomial()
        m = 0

        if (self.__outsample):
            m_start = 1
            fit_formula = f"o ~ {m_start} + "
        else:
            m_start = 2
            fit_formula = f"o ~ I(ge ** {m_start-1}) + "
        
        data = {"o": self.__y, "ge": self.__logit_e}
        inv_chi2 = chi2.ppf(alpha, 1)

        n = m_start
        while n <= maxDeg:
            # Add new term
            fit_formula += f"I(ge ** {n})"

            fit_new = smf.glm(formula=fit_formula, data=data,family=family).fit()

            if n > m_start:
                # Log-likelihood ratio test
                Dm = (2 * (fit.llf - fit_new.llf))

                # Use previous order for m if model does not improve
                if Dm < inv_chi2:
                    m = n-1
                    break

            m = n
            fit = fit_new
            n += 1

            fit_formula += " + "
        return(fit, m)


    def __test(self, alpha):
        # Log-Likelihood of perfectly calibrated model (y_i = p_i)
        llf_perfect = np.sum(xlogy(self.__y, self.__p) + xlogy(1 - self.__y, 1 - self.__p))
        T = 2 * (self.__model.llf - llf_perfect)

        pval = 1 - cdf_m(T, self.__m, self.__outsample, alpha)

        return T, pval


    def _root_fun(self, x, *args):
        m, q, confidence = args
        return cdf_m(x, m, self.__outsample, q) - confidence


    def _fun(self, alpha, *args):
        geM, sign = args
        return sign * (alpha @ geM)


    def _jac(self, alpha, *args):
        geM, sign = args
        return sign * geM


    def __calculate_boundaries(self, confidence, size=50, q=.95, **kwargs):
        # Cache boundaries for each confidence interval to save the parameters used in their calculation.
        if confidence in self.__boundaries:
            params = self.__boundaries[confidence]["params"]

            # If a request for the same interval is requested
            # it is only calculted one of the parameters change.
            # The parameters that modify the belt are (size, q, m).
            # In the case `size` changes, we compute the boundaries
            # only if the new `size` is greater than the previously used.
            
            if (size <= params["size"] and
                    q == params["q"] and
                    self.__m == params["m"]):
                return self.boundaries[confidence]["boundaries"]

        # New parameters
        params = {"size": size, "q": q, "m": self.__m}

        # Find ky
        if self.__outsample:
            ky = (self.__m - 1) * chi2.ppf(q, 1)
        else:
            ky = (self.__m - 2) * chi2.ppf(q, 1)

            try:
                cdf_m(ky, self.__m, self.__outsample, q)
            except:
                ky += 1e-04


        args = self.__m, q, confidence


        k = brentq(self._root_fun, ky, 40, args=args)

        # Calculate logit(E) matrix
        M = np.linspace([0], [self.__m], num = self.__m + 1, axis=1)
        Ge = logit(self.__p)[np.newaxis]
        GeM = Ge.T ** M

        # Upper boundary (Eq27)
        boundary = self.__model.llf - k / 2

        # Create subset based on size
        logit_sub = np.linspace(np.min(Ge), np.max(Ge), num=size//2)
        e_sub = np.linspace(np.min(self.__p), np.max(self.__p), num=size//2)
        Ge_sub = np.sort(np.append(logit_sub, logit(e_sub)))
        GeM_sub = Ge_sub[np.newaxis].T ** M

        # Constraint function (Eq27)
        def fun_lalpha(alpha):
            # Calculate probability
            alphaE = expit(GeM @ alpha)

            # Clip probability to epsilon so
            # we can compute log-likelihood
            eps = 1e-5
            alphaE = np.clip(alphaE, eps, 1-eps)

            # Compute Log-likelihood
            lalpha = xlogy(self.__y, alphaE) + xlogy(1-self.__y, 1 - alphaE)
            return np.nansum(lalpha)

        def jac_lalpha(alpha):
            # Calculate probability
            alphaE = expit(GeM @ alpha)
            return (self.__y - alphaE) @ GeM


        constraints = NonlinearConstraint(
                fun_lalpha,
                boundary, 0,
                jac_lalpha,
                keep_feasible=True
            )

        def _minimize(args):
            return minimize(
                fun=self._fun, x0=self.__model.params, args=args,
                method='trust-constr', jac=self._jac,
                hess=lambda alpha, *args: np.zeros((self.__m+1,)),
                constraints=constraints, tol=1e-4
            ).x

        
        lower, upper = [], []
        for geM in tqdm(GeM_sub):
            
            # Minimize alpha to find lower bound
            args = (geM, 1)
            min_alpha = _minimize(args)

            # Maximize alpha to find upper bound
            args = (geM, -1)
            max_alpha = _minimize(args)

            # Calculate bounds
            lower.append(expit(min_alpha @ geM))
            upper.append(expit(max_alpha @ geM))


        # Save parameters
        boundaries = np.array([expit(Ge_sub), lower, upper]).T
        self.__boundaries[confidence] = {
            "params": params,
            "boundaries": boundaries
        }
        
        return boundaries


    def __get_plot_text(self):
        infos = {"Degree Fit":self.__m, "n":self.__n, "T":self.__T}

        lines = ['{:<10s}{:>8d}'.format("n",self.__n)]
        lines = []
        for k, v in infos.items():
            if (isinstance(v,int)):
                lines.append('{:<10s}{:>8d}'.format(k,v))
            else:
                lines.append('{:<10s}{:>8.3f}'.format(k,v))
        
        # Set text for p-value
        if self.__pval < .001:
            lines.append('{:<10s}{:>8s}'.format("p-value","< 0.001"))
        else:
            lines.append('{:<10s}{:>8.3f}'.format("p-value",self.__pval))

        textstr = '\n'.join(lines)
        return textstr


    # Return teststatistic and p-value
    def stats(self):
        """Get the calibration belt test result, withour drawing the plot.

        Returns
        -------
            T : float
                The Calibration plot test statistic T.
            p : float
                The p-value of the test.

        Notes
        -----
        A low value for the teststatistic and a high p-value (>0.05) indicate a well calibrated model.

        Examples
        --------
        >>> from pycaleva.calbelt import CalibrationBelt
        >>> cb = CalibrationBelt(y_test, pred_prob, outsample=True)
        >>> cb.stats()
        calbelt_result(statistic=1.6111330037643796, pvalue=0.4468347221346196, fig=None)
        """

        return calbelt_result(self.__T, self.__pval, None)



    def plot(self, alpha=.95, **kwargs):
        """Draw the calibration belt plot.
        
        Parameters
        ----------
        alpha: float, optional
            Sets the significance level.
        confLevels: list, optional
            Set the confidence intervalls for the calibration belt.
            Defaults to [0.8,0.95].

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
        CalibrationEvaluator.calbelt
        

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
            Available from https://github.com/lbulgarelli/calibration

        ..  [8] Nattino, G., Finazzi, S., Bertolini, G., Rossi, C., & Carrara, G. (2017).
            givitiR: The giviti calibration test and belt (R package version 1.3) [Computer
            software]. The Comprehensive R Archive Network.
            Available from https://CRAN.R-project.org/package=givitiR
        

        Examples
        --------
        >>> from pycaleva.calbelt import CalibrationBelt
        >>> cb = CalibrationBelt(y_test, pred_prob, outsample=True)
        >>> cb.plot()
        calbelt_result(statistic=1.6111330037643796, pvalue=0.4468347221346196, fig=matplotlib.figure)

        """

        for confidence in self.__confLevels:
            self.__calculate_boundaries(confidence, q=alpha, **kwargs)

        # Plot stats
        fig, ax = plt.subplots(1, figsize=(10,6))

        # Info Textbox upper left
        props = dict(boxstyle='round', facecolor='white', alpha=0.4)
        ax.text(0.00, 0.85, self.__get_plot_text(), fontsize=10, family='monospace', bbox=props)

        # Set primary color belt
        facecol = 'cornflowerblue' 

        # Set alpha of colour for each confidence level
        col_alphas = np.linspace(start=0.9, stop=0.4, num=len(self.__confLevels) )
        
        legend_elements = []

        # Update boundary and legend for each confidence level
        for i,confLevel in enumerate(self.__confLevels):
            legend_elements.append( Patch(facecolor=facecol, alpha=col_alphas[i], label=f'{int(confLevel*100)}%') )
            bound = self.__boundaries[confLevel]["boundaries"].T
            ax.fill_between(bound[0], bound[1], bound[2], color='white', alpha=1)
            ax.fill_between(bound[0], bound[1], bound[2], color=facecol, alpha=col_alphas[i])

        # Set legend
        ax.legend(title='Confidence level', handles=legend_elements, loc='lower right')

        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Actual Probability')

        # Plot perfect calibration using doted line
        ax.plot([0, 1], [0, 1], "k:")

        return calbelt_result(self.__T, self.__pval, fig)