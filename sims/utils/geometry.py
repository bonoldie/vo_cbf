import numpy as np


class Cone:
    """ 
    2D Cone

    Used in VO
    """

    def __init__(self, fulcrum: np.ndarray):
        self.fulcrum = fulcrum
