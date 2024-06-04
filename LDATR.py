# This code was written by Mohammad Jalali in February 2024. It implements a distributed
# optimization algorithm using the Trust Region method, which offers better convergence
# properties compared to the Line Search approach. The architecture utilizes a Master-Worker
# model, where the Master coordinates the optimization process and the Workers work on 
# partitions of the data to compute local objective functions and gradients. Instead of 
# directly computing the Hessian matrix, which requires significant computational resources
# and memory, this implementation leverages the L-BFGS method. L-BFGS approximates the Hessian
# matrix in a more efficient manner, significantly reducing the computational burden and 
# memory requirements. This approach allows for effective and scalable distributed optimization
# across multiple processing units, making it suitable for large-scale machine learning and 
# data analysis tasks.

import numpy as np
import matplotlib.pyplot as plt
import time
from mpi4py import MPI
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score, f1_score


# Define the sigmoid function, which is used in logistic regression models
def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

# Calculate the gradient of the logistic regression cost function
def logistic_regression_gradient(X, y, params):
    """Compute gradient of the logistic regression cost function."""
    predictions = sigmoid(X.dot(params))
    gradient = np.dot(X.T, (predictions - y)) / len(y)
    return gradient


class Master:
    """Master class that controls the optimization process."""
    def __init__(self, size, max_it, learning_rate=0.1):
        # Initialization of the master's parameters and settings
        self.size = size  # Number of processes (slaves + master)
        self.params = None  # Parameters of the model to be optimized
        self.B_inv = None  # Inverse Hessian approximation
        self.u = None  # Approximation of B * s, where B is an approximation of the Hessian
        self.g = None  # Gradient vector
        self.t = 0  # Current iteration
        self.num_iterations = max_it  # Maximum number of iterations
        self.learning_rate = learning_rate  # Learning rate for updates
        self.history = []  # History of parameter updates


    def distribute_data(self, X, y):
        """
        Distributes the data among slave nodes for parallel processing.
        
        This function splits the dataset into chunks and sends each chunk to a different slave node. 
        This is essential for parallelizing the optimization process across multiple processing units.
        
        Parameters:
        - X (numpy.ndarray): The feature matrix for the dataset.
        - y (numpy.ndarray): The labels for the dataset.
        
        The dataset is split based on the number of slave nodes available, ensuring each node receives a roughly equal portion of data.
        """
        print(f"Master is distributing data...")
        chunk_size = len(X) // (self.size - 1)  # Calculate the size of each chunk
        for i in range(1, self.size):
            start = (i - 1) * chunk_size
            end = len(X) if i == self.size - 1 else start + chunk_size
            
            MPI.COMM_WORLD.send(X[start:end], dest=i, tag=0)  # Send a chunk of the feature matrix
            MPI.COMM_WORLD.send(y[start:end], dest=i, tag=1)  # Send the corresponding labels
            print(f"Sending data to Slave {i}: X[{start}:{end}], y[{start}:{end}]")


    def update_and_control(self, X_test, y_test):

        time_histories = {}  # For storing time histories
        f_histories = {}  # For storing the objective function histories from each Slave
        self.f1_scores = []

        while self.t < self.num_iterations:
            #print(f'iteration {self.t} in master`')
            status = MPI.Status()
            p, du, diff_grad, q, alpha, beta = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=1, status=status)
            sending_rank = status.Get_source()
            #print(f"Master received update from Slave {sending_rank}, du:{du}, y:{diff_grad}, q:{q}, alpha:{alpha}, beta:{beta} ")

            self.u = self.u + du
            self.g = self.g + diff_grad
            v = self.B_inv @ diff_grad
            U = self.B_inv - np.outer(v, v) / (alpha + v.T @ diff_grad)
            w = U @ q
            self.B_inv = U + np.outer(w, w) / (beta - q.T @ w)


            self.params = self.params + p
            #self.params = self.B_inv @(self.u - self.learning_rate * self.g)
            
            
            self.history.append(self.params.copy())
            self.t += 1
            predictions = sigmoid(X_test.dot(self.params)) >= 0.5
            f1 = f1_score(y_test, predictions)
            self.f1_scores.append(f1)

            if self.t < self.num_iterations:
                MPI.COMM_WORLD.send(self.params, dest=sending_rank, tag=2)
                #print(f"Master send x={self.x} to Slave {sending_rank}")
            else:
                #print(f'history of x = {self.history}')
                print(f't master = {self.t}')
                print(f'parameter master = {self.params}')
                for r in range(1, self.size):
                    MPI.COMM_WORLD.send(None, dest=r, tag=3)
                print('======end of execution of master======')

        # After the last iteration, receive the objective function histories from Slaves
        
        for _ in range(1, self.size):
            f_history, time_history = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=4, status=status)
            sending_rank = status.Get_source()
            f_histories[sending_rank] = f_history
            time_histories[sending_rank] = time_history

        # Plot the convergence graph
        self.plot_time_convergence(time_histories, f_histories, {i: f"Worker {i}" for i in f_histories.keys()})

    def plot_f1_scores(self):
        plt.plot(self.f1_scores, linewidth=3)
        plt.xlabel('Iteration', fontsize=12, fontweight='bold')
        plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
        plt.title('F1-Score over Iterations', fontsize=14, fontweight='bold')
        plt.grid(True)
        plt.show()


    def plot_time_convergence(self, time_histories, f_histories, labels):
        for proc_id, times in time_histories.items():
            f_values = f_histories[proc_id]
            cumulative_time = np.cumsum(times)  
            plt.plot(cumulative_time, f_values, label=labels.get(proc_id), linewidth=3)

        plt.xlabel("Time (ms)", fontsize=12, fontweight='bold')
        plt.ylabel("Objective Function Value", fontsize=12, fontweight='bold')
        plt.title("Convergence Over Time", fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True)  
        plt.show()


class Slave:
    def __init__(self, rank, mem, lambda_reg=0.01):
        """Initializes a slave node with given parameters."""
        self.lambda_reg = lambda_reg  # Regularization parameter
        self.rank = rank  # MPI rank of the slave
        self.mu = deque([], mem)  # Memory for L-BFGS updates
        self.params = None  # Local parameters
        self.old_params = None  # Parameters from the previous iteration
        self.u = None  # Approximation of B * s, where B is an approximation of the Hessian and s = params - old_params
        self.u_prev = None  # u from the previous iteration
        self.operation_count = 1  # Counter for the number of optimization steps performed by this slave
        self.delta = None  # Radius of the trust region, controlling the step size in optimization
        self.f_history = []  # For storing the history of objective function values


    def receive_data(self):
        """
        Receives the data chunk assigned to this slave node from the master node.
        
        This function is called by each slave node to receive its portion of the dataset for local processing. 
        The data is sent by the master node and includes both features (X) and labels (y) for the subset of the dataset assigned to this slave.
        
        Upon receiving the data, the slave node initializes its local parameters for optimization. These parameters include:
        - params: The optimization parameters, initialized to zeros based on the number of features in the received dataset.
        - old_params: A copy of the initial parameters for tracking changes between iterations.
        - u: An approximation of the product of the inverse Hessian matrix and the gradient, initialized to zeros.
        - u_prev: A copy of the initial 'u' values for use in L-BFGS updates.
        
        The shape of the received dataset is printed to the console for verification.
        """
        print(f"Slave {self.rank} is receiving data...")
        self.X = MPI.COMM_WORLD.recv(source=0, tag=0)  # Receives the feature matrix from the master node
        self.y = MPI.COMM_WORLD.recv(source=0, tag=1)  # Receives the label vector from the master node
        print(f"Slave {self.rank} received data: X.shape={self.X.shape}, y.shape={self.y.shape}")
        
        # Initialize parameters based on the number of features in the received dataset
        self.params = np.zeros(self.X.shape[1])
        self.old_params = np.copy(self.params)
        self.u = np.zeros(self.X.shape[1])
        self.u_prev = np.zeros(self.X.shape[1])



    def f(self, x):
        reg_term = (self.lambda_reg / (2 * len(self.y))) * np.sum(x**2)
        return np.mean(-self.y * np.log(sigmoid(self.X @ x)) - (1 - self.y) * np.log(1 - sigmoid(self.X @ x))) + reg_term

    def grad(self, params):
        reg_term = self.lambda_reg * params / len(self.y)
        return logistic_regression_gradient(self.X, self.y, params) + reg_term

    
    def lbfgs(self, gamma, mu, params):
        u = gamma * params
        for y, q, alpha, beta in mu:
            c1 = y.T @ params / alpha
            c2 = q.T @ params / beta
            u = u + c1 * y - c2 * q
        return u


    def take_one_step_of_GD(self):
        # In the first iteration, each slave takes one step of gradient descent for better initialization.
        # This step is necessary because the L-BFGS calculations in subsequent iterations rely on having two sets of parameter vectors:
        # one from the current iteration ('self.params') and one from the previous iteration ('self.old_params').
        # Since each slave initially receives only one vector of parameters from the master, performing one step of gradient descent
        # provides the needed second vector. This ensures that 'self.old_params' can be set to the parameters received from the master,
        # and 'self.params' gets updated to reflect the new position after the gradient descent step.
        # This process facilitates the L-BFGS method to start functioning properly from the second iteration onwards.
        self.params = self.params - 0.001 * self.grad(self.params)  
        self.u = self.u + self.params  # Assuming the initial Hessian is the identity matrix for simplicity.

      
    def operate(self):
        delta_max = 10.0  
        epsilon = 1e-6
        time_history = []  # For storing time history
        while True:
            start_time = time.time() * 1000
            s = self.params - self.old_params
            #print(f"===slave{self.rank}===>>x={self.x}, z={self.z}, s={s}")

            grad_params = self.grad(self.params)
            grad_old_params = self.grad(self.old_params)

            diff_grad = grad_params - grad_old_params

            dot_sy = np.dot(s, diff_grad)
            gamma = np.dot(diff_grad, diff_grad) / dot_sy #if dot_sy != 0 else 1.0

            q = self.lbfgs(gamma, self.mu, s)

            alpha = np.dot(diff_grad, s)
            beta = np.dot(s, q)

            self.mu.append((diff_grad, q, alpha, beta))

            u = self.lbfgs(gamma, self.mu, self.params)

            du = u - self.u_prev

            # Here, we are constructing a second-order model (m) for the objective function based on Taylor expansion.
            # The model 'm' is defined as follows: m = f(x) + grad_f(x)^T * p + 0.5 * p^T * B * p
            # Instead of directly using the Hessian matrix (B), we use its approximation through 'u'.
            # 'u' approximates the B * s, where B is an approximation of the Hessian and s = params - old_params
            # so the model 'm' is defined as follows: m = f(x) + grad_f(x)^T * p + 0.5 * p^T * u + 0.5 * p^T * p
            # Since 'B' is approximated by 'u', the formula is adapted to use 'u' instead of the full Hessian matrix.
            p = -0.5 * u - grad_params # Calculating the step direction 'p'
            # Calculating the actual value of the objective function at the current parameters 'self.params'.
            f_x = self.f(self.params)

            # Calculating the predicted value of the objective function after taking the step 'p'.
            f_x_p = self.f(self.params + p)

            # 'rho' measures the ratio between the actual reduction in the objective function and the predicted reduction.
            # A high value of 'rho' indicates that the model (m) accurately predicts the change in the objective function,
            # while a low or negative value suggests the model is not accurate.
            rho = (f_x - f_x_p) / (-grad_params.T @ p - 0.5 * p.T @ u - 0.5 * np.dot(p, p))
            
            if self.operation_count == 1:
                # This initialization ensures that the direction is within the trust region for the first step.
                # We set the delta based on the initial step size to make sure that subsequent updates 
                # can refine the delta value for better optimization direction.
                self.delta = 2 * np.linalg.norm(p)

            # Calculate the norm of the proposed step vector 'p'
            norm_p = np.linalg.norm(p)
            # Adjust the trust region radius 'delta' based on the ratio 'rho'
            # If 'rho' is less than 0.25, it indicates that the proposed step 'p' did not result in a sufficient decrease in the objective function,
            # so the trust region radius is reduced to make the next step more conservative
            if rho < 0.25:
                self.delta /= 4
            # If 'rho' is greater than 0.75 and the norm of 'p' is approximately equal to 'delta', it suggests that the step 'p' was effective,
            # and the trust region could be expanded. However, 'delta' is capped by 'delta_max' to prevent it from becoming too large.
            elif rho > 0.75 and np.abs(norm_p - self.delta) < epsilon:
                self.delta = min(2 * self.delta, delta_max)

            # Adjust the step 'p' if its norm is greater than the trust region radius 'delta'
            # This ensures that the step 'p' does not exceed the bounds of the trust region.
            # The step 'p' is scaled down to match exactly the trust region radius if it's originally larger than 'delta'.
            # This scaling helps maintain the direction of 'p' while adjusting its magnitude to stay within the trust region.
            if norm_p < self.delta:
                p = p  # If 'p' is within the trust region, no adjustment is needed
            else:
                p = p * (self.delta / norm_p)  # Scale down 'p' to fit within the trust region
                        
            # Save the current value of the objective function
            self.f_history.append(f_x)
            
            MPI.COMM_WORLD.send((p, du, diff_grad, q, alpha, beta), dest=0, tag=1)
            
            new_params = MPI.COMM_WORLD.recv(source=0, tag=MPI.ANY_TAG)
            #print(f"Slave {self.rank} received x={new_params} from master")

            end_time = time.time() * 1000  
            time_history.append(end_time - start_time)

            if new_params is None:
                MPI.COMM_WORLD.send((self.f_history, time_history), dest=0, tag=4)
                break
            self.old_params = self.params
            self.params = new_params
            self.u_prev = u # save (u = Bi * x) for next iteration
            self.operation_count += 1

            

def calculate_metrics(X, y, params):
    """
    Calculate and print performance metrics for the logistic regression model.

    Parameters:
    - X: The feature matrix for the test dataset.
    - y: The true labels for the test dataset.
    - params: The parameters (weights) of the logistic regression model.

    This function calculates the balanced accuracy and F1-Score for the model's predictions.
    Balanced accuracy is the average of recall obtained on each class, useful for imbalanced datasets.
    F1-Score is the harmonic mean of precision and recall, providing a balance between them.

    Returns:
    - balanced_acc: The balanced accuracy of the model's predictions.
    - f1: The F1-Score of the model's predictions.
    """
    # Generate predictions by applying the sigmoid function to the dot product of X and params
    predictions = sigmoid(X.dot(params)) >= 0.5
    
    # Calculate balanced accuracy
    balanced_acc = balanced_accuracy_score(y, predictions)
    
    # Calculate F1-Score
    f1 = f1_score(y, predictions)

    # Print the calculated metrics
    print(f"Balanced Accuracy: {balanced_acc:.2f}")
    print(f"F1-Score: {f1:.2f}")

    return balanced_acc, f1



def main():
    """
    The main function that initializes the MPI environment, loads and preprocesses the dataset,
    then executes the distributed optimization process using a master-slave architecture.
    """

    comm = MPI.COMM_WORLD  # Initialize the MPI environment
    rank = comm.Get_rank()  # Get the rank of the current process
    size = comm.Get_size()  # Get the total number of processes

    mem = 20  # Define the memory budget for limited memory BFGS (L-BFGS) updates
    max_it = 50  # Set the maximum number of iterations for the optimization process
    
    if rank == 0:
        # This block is executed only by the master process.
        
        master = Master(size, max_it)  # Initialize the master node with the total number of processes and max iterations
        
        # Load Breast Cancer dataset
        data = load_breast_cancer()
        X = data.data  # Feature matrix
        y = data.target  # Labels
        print("Master is loading and preprocessing the Breast Cancer dataset...")


        # Preprocess the data before distribution and optimization
        scaler = StandardScaler() # Initialize a standard scaler
        X = scaler.fit_transform(X) # Normalize the feature matrix to have a mean of 0 and a standard deviation of 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # Split the dataset into training and testing sets

        print(f"Data loaded: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
        print(f"Data loaded: X_test.shape={X_test.shape}, y_test.shape={y_test.shape}")


        master.params = np.random.rand(X_train.shape[1])  # Random initial parameters
        #master.params = np.zeros(X_train.shape[1]) #or use this for better initialization
        master.B_inv = np.eye(X_train.shape[1])
        master.u = np.zeros(X_train.shape[1])
        master.g = np.zeros(X_train.shape[1])

        master.distribute_data(X_train, y_train)
        master.update_and_control(X_test, y_test)
        master.plot_f1_scores()

        # After the optimization, evaluate the model on the test set
        balanced_acc, f1 = calculate_metrics(X_test, y_test, master.params)
        print(f"Final parameters: {master.params}")
        print(f"Balanced Accuracy on test data: {balanced_acc:.2f}")
        print(f"F1-Score on test data: {f1:.2f}")
        
    else:
        # This block is executed by the slave processes.

        lambda_reg = 0.01  # Set the regularization parameter
        slave = Slave(rank, mem, lambda_reg)  # Initialize a slave node
        
        slave.receive_data()  # Receive a chunk of data from the master process
        slave.take_one_step_of_GD()  # Perform an initial gradient descent step for better parameter initialization
        slave.operate()  # Start the optimization process on the received chunk of data

if __name__ == "__main__":
    main()
