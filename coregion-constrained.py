import tensorflow as tf 
from gpflow.kernels import Kernel
from gpflow import Parameter


class ConstrainedCoregion(Kernel):
    """
    A Coregionalization kernel. The inputs to this kernel are _integers_ (we
    cast them from floats as needed) which usually specify the *outputs* of a
    Coregionalization model.

    The kernel function is an indexing of a positive-definite matrix:

      K(x, y) = B[x, y] .

    To ensure that B is positive-definite, it is specified by the two
    parameters of this kernel, W and kappa:

      B = W Wᵀ + diag(kappa) .

    We refer to the size of B as "output_dim x output_dim", since this is the
    number of outputs in a coregionalization model. We refer to the number of
    columns on W as 'rank': it is the number of degrees of correlation between
    the outputs.

    NB. There is a symmetry between the elements of W, which creates a local
    minimum at W=0. To avoid this, it is recommended to initialize the
    optimization (or MCMC chain) using a random W.
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
        self.W = Parameter(W)
        self.kappa = Parameter(kappa, transform=positive())

    @check_shapes(
        "return: [P, P]",
    )
    def output_covariance(self) -> tf.Tensor:
        B = tf.linalg.matmul(self.W, self.W, transpose_b=True) + tf.linalg.diag(self.kappa)
        return B

    @check_shapes(
        "return: [P]",
    )
    def output_variance(self) -> tf.Tensor:
        B_diag = tf.reduce_sum(tf.square(self.W), 1) + self.kappa
        return B_diag

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

