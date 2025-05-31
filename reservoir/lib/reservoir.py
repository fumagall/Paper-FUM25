from typing import Literal
from numpy import ones, matrix, array, eye, tanh, ndarray, empty
from numpy.linalg import inv, pinv


class JAU24a:
    def __init__(self, weights_internal: matrix, weights_input: matrix, k: float=0.4, g: float=0.4):
        """A reservoir of the form tanh(k * weights_internal @ state + g * weights_input @ input).

        :param matrix weights_internal: a n_nodes x n_nodes matrix.
        :param matrix weights_input: a n_nodes x system_dimension matrix.
        :param float k: Scalar for weights_internal, defaults to 0.4
        :param float g: Scalar for weights_input, defaults to 0.4
        """
        self.weights_internal = weights_internal
        self.weights_input = weights_input
        self.k, self.g = k, g

    @staticmethod
    def _add_bias_from_array(array: ndarray) -> ndarray:
        """Extends the array by one to add a 1 as bias.

        :param ndarray array: Any 1d array
        :return ndarray: array with bias
        """
        out = ones((*array.shape[:-1], array.shape[-1] + 1))
        out[..., :-1] = array
        return out

    @staticmethod
    def predict(reservoir_responses: ndarray, weights_with_bias: ndarray) -> ndarray:
        """Calculates the prediction of the reservoir using the weights with bias.

        :param ndarray reservoir_responses: A responds from a reservoir.
        :param ndarray weights_with_bias: Weights with bias from regression problem.
        :return ndarray: The prediction, the reservoir produces.
        """
        return reservoir_responses @ weights_with_bias[:-1] + weights_with_bias[-1]

    def _reservoir_step(self, current_reservoir_state: ndarray, driving_input: ndarray) -> ndarray:
        """A function that calculates a single the reservoir respons, while driving it with an input.

        :param ndarray current_reservoir_state: A single reservoir state, representing the current state.
        :param ndarray driving_input: A single input that is use to drive the system.
        :return ndarray: The reservoir response.
        """
        return tanh(self.k * self.weights_internal @ current_reservoir_state + self.g * self.weights_input @ driving_input)

    def driven_response(self, initial_reservoir_state: ndarray, driving_inputs: ndarray) -> ndarray:
        """A function that calculates the reservoir responses, while driving it with an input.

        :param ndarray initial_reservoir_state: The state in which the reservoir starts.
        :param ndarray driving_inputs: A list of arrays. Each array is used as input to drive the reservoir.
        :return ndarray: The reservoir responses.
        """
        states = empty((len(driving_inputs), *initial_reservoir_state.shape))

        # run training
        states[0] = self._reservoir_step(initial_reservoir_state, driving_inputs[0])

        for i, input in enumerate(driving_inputs[1:]):
            states[i + 1] = self._reservoir_step(states[i], input)

        return states

    def run_and_predict_autonomous_system(self, inital_reservoir_state: ndarray, weights_with_bias: ndarray, n_timesteps: int) -> tuple[ndarray, ndarray]:
        """Runs the autonomous system and returns its internal responses and predictions.

        :param ndarray inital_reservoir_state: An array with length n_nodes representing the initial reservoir state
        :param wndarray eights_with_bias:  Trained weights with the last weight being 1 to incoperate the bias
        :param int n_timesteps: Number of timesteps for which the autonomous system should run
        :return list[ndarray, ndarray]: Reservoir response and prediction.
        """
        states = empty((n_timesteps, *inital_reservoir_state.shape))

        states[0] = inital_reservoir_state

        for t in range(n_timesteps - 1):
            states[t + 1] = self._reservoir_step(states[t], self.predict(states[t], weights_with_bias))

        return states, self.predict(states, weights_with_bias)
    
    @classmethod
    def linear_regression_inv(cls, reservoir_responses: matrix, target: ndarray, regulazation_lambda: float=0) -> ndarray:
        """Using numpy inv, calculates the weights to ridge regression problem min_w ||target - responses @ w|| + lambda ||w||^2.

        :param matrix reservoir_responses: The responses of a reservoir.
        :param ndarray target: The target that should be reproduced linearly from the responses.
        :param float regulazation_lambda: Regulatizer, defaults to 0
        :return ndarray: Weight for linear regression
        """
        S = cls._add_bias_from_array(reservoir_responses)
        return inv(S.T @ S + regulazation_lambda * eye(S.shape[1])) @ S.T @ target

    @classmethod
    def linear_regression_pinv(cls, reservoir_responses: matrix, target: ndarray) -> ndarray:
        """Using numpy pinv, calculates the weights to regression problem min_w ||target - responses @ w||.

        :param matrix reservoir_responses: The responses of a reservoir.
        :param ndarray target: The target that should be reproduced linearly from the responses.
        :return ndarray: Weight for linear regression
        """
        S = cls._add_bias_from_array(reservoir_responses)
        return pinv(S) @ target 