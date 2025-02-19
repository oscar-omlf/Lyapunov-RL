from typing import Tuple

class Welford:
    """
    Implements Welford's online algorithm for computing the running mean and variance.
    
    This class updates its internal state each time a new sample is added, and can provide
    the current mean and variance without storing all the data points.
    """
    def __init__(self):
        """
        Initialize the Welford aggregator.
        
        Attributes:
            count (int): Number of samples seen so far.
            mean (float): Running mean of the samples.
            M2 (float): Sum of squares of differences from the current mean.
        """
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update_aggr(self, new_val: float) -> None:
        """
        Update the aggregator with a new value.
        
        This method updates the running count, mean, and the aggregated sum of squared differences (M2)
        based on the new value provided.
        
        Args:
            new_val (float): The new sample to incorporate into the running statistics.
        """
        self.count += 1
        delta = new_val - self.mean
        self.mean += delta / self.count
        delta2 = new_val - self.mean
        self.M2 += delta * delta2

    def get_curr_mean_variance(self) -> Tuple[float, float]:
        """
        Retrieve the current mean and variance of the samples seen so far.
        
        This method calls _finalize_aggr() to compute the current statistics and returns
        the mean and the variance.
        
        Returns:
            Tuple[float, float]: A tuple containing the current mean and variance.
        """
        mean, _, var = self._finalize_aggr()
        return mean, var

    def _finalize_aggr(self) -> Tuple[float, float, float]:
        """
        Compute the final aggregated statistics: mean, variance, and sample variance.
        
        For fewer than two samples, variance and sample variance are returned as 0.
        Otherwise, the variance is computed as M2 divided by the count, and the sample variance
        is computed as M2 divided by (count - 1).
        
        Returns:
            Tuple[float, float, float]: A tuple containing the mean, variance, and sample variance.
        """
        if self.count < 2:
            return self.mean, 0, 0
        else:
            variance = self.M2 / self.count
            sample_variance = self.M2 / (self.count - 1)
            return self.mean, variance, sample_variance