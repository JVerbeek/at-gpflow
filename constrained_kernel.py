from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from check_shapes import check_shape as cs
from check_shapes import check_shapes, inherit_check_shapes

from gpflow.base import AnyNDArray, Parameter, TensorType
from gpflow.utilities import positive, to_default_float
from gpflow.kernels.base import ActiveDims, Kernel
import gpflow


class ConstrainedCoregion(Kernel):
    """
    Coregionalization kernel, but coregion matrix is forced to contain values between -1 and 1. 
    """

    def __init__(
        self,
        output_dim: int,
        rank: int,
        *,
        active_dims: Optional[ActiveDims] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        :param output_dim: number of outputs expected (0 <= X < output_dim)
        :param rank: number of degrees of correlation between outputs
        """

        # assert input_dim == 1, "Coregion kernel in 1D only"
        super().__init__(active_dims=active_dims, name=name)

        self.output_dim = output_dim
        self.rank = rank
        W: AnyNDArray = 0.1 * np.ones((self.output_dim, self.rank))
        kappa = np.ones(self.output_dim)
        self.A = gpflow.Parameter(0.9999, transform=tfp.bijectors.SoftClip(tf.cast(-1, tf.float64), tf.cast(1, tf.float64)))
        self.B = gpflow.Parameter(0.1, transform=tfp.bijectors.SoftClip(tf.cast(-1, tf.float64), tf.cast(1, tf.float64)))

    @check_shapes(
        "return: [P, P]",
    )
    def output_covariance(self) -> tf.Tensor:
        C = tf.stack([tf.stack([self.A, self.B]), tf.stack([self.B, self.A]),])
        return C

    @check_shapes(
        "return: [P]",
    )
    def output_variance(self) -> tf.Tensor:
        C = tf.stack([tf.stack([self.A, self.B]), tf.stack([self.B, self.A]),])
        return tf.linalg.diag_part(C)

    @inherit_check_shapes
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        cs(X, "[batch..., N, 1]  # The `Coregion` kernel requires a 1D input space.")

        B = cs(self.output_covariance(), "[O, O]")
        X = cs(tf.cast(X[..., 0], tf.int32), "[batch..., N]")
        if X2 is None:
            batch = tf.shape(X)[:-1]
            N = tf.shape(X)[-1]
            O = tf.shape(B)[-1]

            result = cs(tf.gather(B, X), "[batch..., N, O]")
            result = cs(tf.reshape(result, [-1, N, O]), "[flat_batch, N, O]")
            flat_X = cs(tf.reshape(X, [-1, N]), "[flat_batch, N]")
            result = cs(tf.gather(result, flat_X, axis=2, batch_dims=1), "[flat_batch, N, N]")
            result = cs(tf.reshape(result, tf.concat([batch, [N, N]], 0)), "[batch..., N, N]")
        else:
            X2 = cs(tf.cast(X2[..., 0], tf.int32), "[batch2..., N2]")

            rank2 = tf.rank(X2)

            result = cs(tf.gather(B, X2), "[batch2..., N2, O]")
            result = cs(
                tf.transpose(result, tf.concat([[rank2], tf.range(rank2)], 0)), "[O, batch2..., N2]"
            )
            result = cs(tf.gather(result, X), "[batch..., N, batch2..., N2]")

        return result

    @inherit_check_shapes
    def K_diag(self, X: TensorType) -> tf.Tensor:
        cs(X, "[batch..., N, 1]  # The `Coregion` kernel requires a 1D input space.")

        X = tf.cast(X[..., 0], tf.int32)
        B_diag = self.output_variance()
        return tf.gather(B_diag, X)