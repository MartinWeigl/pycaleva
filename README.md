[![](./misc/logo.svg)](https://martinweigl.github.io/pycaleva/)

pycaleva
====
A framework for calibration measurement of binary probabilistic models.  
  

<img src="./misc/design.png" width="600">


[Documentation]

[Documentation]: https://martinweigl.github.io/pycaleva/

Installation
------------

    $ pip install pycaleva


Usage
-----

- Import and initialize  
    ```python
    from pycaleva import CalibrationEvaluator
    ce = CalibrationEvaluator(y_test, pred_prob, outsample=True, n_groups='auto')
    ```
- Apply statistical tests
    ```python
    ce.hosmerlemeshow()     # Hosmer Lemeshow Test
    ce.pigeonheyse()        # Pigeon Heyse Test
    ce.z_test()             # Spiegelhalter z-Test
    ce.calbelt(plot=False)  # Calibrationi Belt (Test only)
    ```
- Show calibration plot
    ```python
    ce.calibration_plot()
    ```
- Show calibration belt
    ```python
    ce.calbelt(plot=True)
    ```
- Get various metrics
    ```python
    ce.metrics()
    ```

See  the [documentation] of single methods for detailed usage examples.


Features
--------
* Statistical tests for binary model calibration
    * Hosmer Lemeshow Test
    * Pigeon Heyse Test
    * Spiegelhalter z-test
    * Calibration belt
* Graphical represantions showing calibration of binary models
    * Calibration plot
    * Calibration belt
* Various Metrics
    * Scaled Brier Score
    * Adaptive Calibrationi Error
    * Maximum Calibration Error
    * Area within LOWESS Curve
    * (AUROC)

The above features are explained in more detail in PyCalEva's [documentation]