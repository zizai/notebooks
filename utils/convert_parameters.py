import numpy as np


def parameters_to_vector(parameters):
    r"""Convert parameters to one vector

    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    vec = []
    for param in parameters:
        vec.append(param.data)
    return np.concatenate(vec)


def vector_to_parameters(vec, parameters):
    r"""Convert one vector to the parameters

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    # Ensure vec of type Tensor
    if not isinstance(vec, np.ndarray):
        raise TypeError('expected numpy.ndarray, but got: {}'
                        .format(vec.__name__))

    # Pointer for slicing the vector for each parameter
    pointer = 0
    for param in parameters:
        # The length of the parameter
        num_param = param.size
        # Slice the vector, reshape it, and replace the old data of the parameter
        param.data = vec[pointer:pointer + num_param]

        # Increment the pointer
        pointer += num_param
