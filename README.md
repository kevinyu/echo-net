# echo-net

Implementation of dynamic recurrent neural networks as described in Herbert Jaeger, et al. 2004.

## setup

```
$ virtualenv env -p python3        # initialize virtualenv to keep python dependencies separate
$ source env/bin/activate          # activate virtualenv
$ pip install -r requirements.txt  # install python requirements
```

## examples

Initializing network

```
>>> import numpy as np
>>> from echo.network import ReservoirNetwork, ReservoirState
>>> res = Reservoir(800, N_in=1, N_out=1, sparsity=0.25, g=1.8, noise=0.01)
>>> net = ReservoirNetwork(res)
```

Running without input

```
>>> null_input = np.zeros((1, 5000))
>>> result = net.run(input_data=null_input)
>>> result["z"]
```

Learn a sequence

```
>>> t = np.linspace(0, 100.0, 5000)
>>> sine = np.sin(2 * np.pi * 0.1 * t)
>>> net.teach(input_data=sine, target=sine)
>>> result = net.run(input_data=sine)
>>> result["z"]
```
