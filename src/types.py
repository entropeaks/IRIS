from scipy.sparse import csr_matrix
from numpy import ndarray
from typing import TypeAlias, Any

Matrix: TypeAlias = csr_matrix | ndarray
Feature: TypeAlias = int | float | list[Any]