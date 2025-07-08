import gpflow
from gpflow.likelihoods import Likelihood
from typing import Optional
import tensorflow as tf
from gpflow.likelihoods import ScalarLikelihood

class TransferLikelihood(ScalarLikelihood):
    """Copiloted the boilerplate for a custom likelihood class in GPflow."""
    def __init__(self, source: Likelihood, target: Likelihood, **kwargs):
        super().__init__(**kwargs)
        self.source = source
        self.target = target
        self.likelihoods = [source, target]

    def _partition_and_stitch(self, args, func_name: str) -> tf.Tensor:
        """
        args is a list of tensors, to be passed to self.likelihoods.<func_name>

        args[-1] is the 'Y' argument, which contains the indexes to self.likelihoods.

        This function splits up the args using dynamic_partition, calls the
        relevant function on the likelihoods, and re-combines the result.
        """
        # get the index from Y
        args_list = list(args)
        Y = args_list[-1]
        ind = Y[..., -1]
        ind = tf.cast(ind, tf.int32)
        Y = Y[..., :-1]
        args_list[-1] = Y

        # split up the arguments into chunks corresponding to the relevant likelihoods
        args_chunks = zip(*[tf.dynamic_partition(X, ind, len(self.likelihoods)) for X in args_list])

        # apply the likelihood-function to each section of the data
        funcs = [getattr(lik, func_name) for lik in self.likelihoods]
        results = [f(*args_i) for f, args_i in zip(funcs, args_chunks)]

        # stitch the results back together
        partitions = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind, len(self.likelihoods))
        results = tf.dynamic_stitch(partitions, results)

        return results

    def _scalar_log_prob(self, X, F, Y) -> tf.Tensor:
        return self._partition_and_stitch([X, F, Y], "_scalar_log_prob")

    def _predict_log_density(
        self, X, Fmu, Fvar, Y
    ) -> tf.Tensor:
        return self._partition_and_stitch([X, Fmu, Fvar, Y], "predict_log_density")


    def _variational_expectations(
        self, X, Fmu, Fvar, Y
    ) -> tf.Tensor:
        return self._partition_and_stitch([X, Fmu, Fvar, Y], "variational_expectations")


    def _predict_mean_and_var(
        self, X, Fmu, Fvar
    ):
        mvs = [lik.predict_mean_and_var(X, Fmu, Fvar) for lik in self.likelihoods]
        mu_list, var_list = zip(*mvs)
        mu = tf.concat(mu_list, axis=1)
        var = tf.concat(var_list, axis=1)
        return mu, var


    def _conditional_mean(self, X, F) -> tf.Tensor:
        raise NotImplementedError


    def _conditional_variance(self, X, F) -> tf.Tensor:
        raise NotImplementedError