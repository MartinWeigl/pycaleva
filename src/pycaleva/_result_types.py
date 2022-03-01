from collections import namedtuple

# DEFINE RESULT TYPES AS namedtuple

__all__ = ['hltest_result', 'phtest_result', 'ztest_result', 'calbelt_result']

hltest_result = namedtuple('hltest_result', ['statistic', 'pvalue', 'dof'])
phtest_result = namedtuple('phtest_result', ['statistic', 'pvalue', 'dof'])
ztest_result = namedtuple('ztest_result', ['statistic', 'pvalue'])
calbelt_result = namedtuple('calbelt_result', ['statistic', 'pvalue', 'fig'])