#!/usr/bin/env python
#
# pyQvarsi, statistics Module.
#
# Module to compute statistics either from the flow
# or using generic correlations.
#
# Last rev: 16/02/2021

__VERSION__ = 3.1

from .basic           import covariance, R, R2, bias, MSE, RMSE, MAE, RMAE, AME, RAE
from .accumulators    import addS1, addS2, addS3
from .flow_statistics import tripleCorrelation, reynoldsStressTensor, strainTensor, vorticityTensor, TKE, dissipation, taylorMicroscale, kolmogorovLengthScale, kolmogorovTimeScale
from .budgets         import convectionBudget, productionBudget, turbulentDiffusion1Budget, turbulentDiffusion2Budget, molecularDiffusionBudget, pressureStrainBudget, dissipationBudget

del basic, accumulators, flow_statistics, budgets