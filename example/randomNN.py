import numpy as np
from typing import List, Union, Optional

class randomNN:
    def __init__(self, 
                 n_input: int, 
                 n_output: int, 
                 hidden_layers: Optional[List[int]] = None, 
                 activation_functions: Optional[List[str]] = None):
        self.n_input = n_input
        self.n_output = n_output
        
        # Determine the minimum and maximum number of nodes in hidden layers based on input and output dimensions
        min_node = int(np.clip(np.log2((n_input+n_output)**0.2), a_min=4, a_max=11))
        max_node = int(np.clip(np.log2((n_input+n_output)), a_min=min(min_node+3,10), a_max=12))
        
        # Initialize hidden_layers if not provided with random values within a specified range
        if hidden_layers is None:
            hidden_layers = [2**np.random.randint(min_node, max_node) for _ in range(np.random.randint(4, 7))]  # 4, 5, or 6 layers
        
        # Calculate the total number of layers and layer dimensions
        self.n_layers: int = len(hidden_layers) + 2
        self.layer_dims: List[int] = [n_input] + hidden_layers + [n_output]
        
        # Initialize activation functions if not provided with random choices for hidden layers and None for the output layer
        if activation_functions is None:
            activation_functions = [np.random.choice(['elu', 'sin', 'cos', 'tanh', 'sinc']) for i in range(self.n_layers-1)]
            activation_functions.append(None)  # no activation on the last layer
        
        # Store layer activation functions and initialize network parameters
        self.activation_functions: List[Union[str, None]] = activation_functions
        self.parameters: dict = self.initialize_parameters()
        
        # Generate random inputs for normalization calculation
        self.mean_output: float = 0
        self.std_output: float = 1
        random_inputs = np.random.randn(1024, n_input)
        random_outputs = self(random_inputs)
        self.mean_output = np.mean(random_outputs, axis=0)
        self.std_output = np.std(random_outputs, axis=0)
        

    def initialize_parameters(self) -> dict:
        # Initialize weights and biases for each layer using He initialization
        parameters = {}
        for l in range(1, self.n_layers):
            scale_weights = np.sqrt(2.0 / self.layer_dims[l - 1])  # He initialization for weights
            parameters[f'W{l}'] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * scale_weights

            # Initialize biases with small random values
            scale_biases = np.sqrt(0.5 / self.layer_dims[l])
            parameters[f'b{l}'] = np.random.randn(self.layer_dims[l], 1) * scale_biases

        return parameters
    
    def normalize_output(self, output: np.ndarray) -> np.ndarray:
        # Normalize output based on mean and standard deviation
        return (output - self.mean_output) / self.std_output

    def activate(self, Z: np.ndarray, activation_function: Union[str, None]) -> np.ndarray:
        # Apply activation functions based on the specified function
        if activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif activation_function == 'tanh':
            return np.tanh(Z)
        elif activation_function == 'relu':
            return np.maximum(0, Z)
        elif activation_function == 'elu':
            return np.where(Z > 0, Z, np.exp(Z) - 1)
        elif activation_function == 'sin':
            return np.sin(Z)
        elif activation_function == 'cos':
            return np.cos(Z)
        elif activation_function == 'sinc':
            return np.sinc(Z)
        else:
            return Z

    def __call__(self, X: np.ndarray) -> np.ndarray:
        
        X = np.array(X)
        assert X.shape[1] == self.n_input
        # Perform forward propagation through the neural network
        A = X.T  # Transpose the input to make it compatible with matrix multiplication
        for l in range(1, self.n_layers):
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']
            Z = np.dot(W, A) + b
            A = self.activate(Z, self.activation_functions[l])
            
        # Transpose the result back to (n_batch, n_output) and then normalize
        return self.normalize_output(A.T)