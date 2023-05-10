# STROBE and BICORN

This is an implementation of STROBE and BICORN in Python, using asyncio.

Strobe: The algorithm allows a group of nodes to share a secret among themselves in a secure and distributed manner. Each node only knows a part of the secret, and the secret can only be reconstructed if enough nodes collaborate.

Bicorn: A group of nodes each generate their random values from a provided generator and publish their commitments. After revealing their random values, DRB output similar to classic commit-reveal protocol is produced if values are honest, else recovery with t steps of sequential work is required. 

### Requirements
    - Python 3.11 or higher
    - asyncio, sympy, math, socket, json, tk Python libraries

Create a virtual environment and activate it:
```bash
python3 -m venv venv
source venv/bin/activate
```

You can install the required libraries using pip:
```bash
pip install asyncio sympy math socket json tk
```

### Usage

To run the simulation, run the following command:

```
python main.py arg1 arg2 arg3 arg4

arg1: number of total nodes (honest + malicious)
arg2: number of malicious nodes
arg3: 
- in Strobe: t in the t-of-n reconstruction scheme (#nodes needed for reconstruction)
- in Bicorn: time delay t
arg4: the implementation name "strobe" or "bicorn"
```

This will start the simulation with these parameters. The GUI shows the current round, message, amount of data transferred, time taken etc shared by each node in that round.

### Implementation details

Strobe:

Each node generates a random polynomial of degree n-1 with a secret x_0 as the constant term, and shares the polynomial coefficients with the other nodes. Each node then evaluates the polynomial at its own ID to get its own share of the secret.

In each round, each node sends its current share to a random subset of other nodes. Once a node has received shares from at least t+1 nodes, it can reconstruct the polynomial coefficients using Lagrange interpolation, and hence the secret. The nodes then increment the round counter and start a new round.

<strong>The algorithm is secure as long as the number of compromised nodes is less than t.</strong>

Bicorn:

At setup, a provided generator g and h = g^2^t is generated. Each node generates a random alpha value from uniform distribution B based on g, and publishes the commitment. After all alpha values are revealed, each node constructs the polynomial by checking the alpha against the published commitment - if verified it will take the product of h^alpha, else it will recover with c^2^t. 
