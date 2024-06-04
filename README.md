# L-DATR: A Limited-Memory Distributed Asynchronous Trust-Region Method

This is the implementation of the paper **L-DATR: A Limited-Memory Distributed Asynchronous Trust-Region Method**. The code provides a distributed optimization algorithm using the Trust Region method, leveraging the L-BFGS method for efficient Hessian approximation. This approach is suitable for large-scale machine learning and data analysis tasks.

## Requirements

- numpy
- matplotlib
- mpi4py
- sklearn

## Usage

To run the implementation, you can use the following command:

```bash
mpirun -n 4 python LDATR.py
