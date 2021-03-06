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
import numpy as np
import matplotlib.pyplot as plt
from echo.network import ReservoirNetwork, Reservoir

res = Reservoir(800, N_in=1, N_out=1, sparsity=0.25, g=1.8, noise=0.01)
net = ReservoirNetwork(res)
```

Running without input

```
null_input = np.zeros((1, 10000))
result = net.run(input_data=null_input)

# Plot activity of some units
plt.figure(1)
plt.plot(result["r"][:, :5])

# Plot the output activity
plt.figure(2)
plt.plot(result["z"][0])
```
![Null input example](/images/null_input_example.png?raw=true)

Learn a sequence

```
t = np.linspace(0, 100.0, 5000)
sine = np.sin(2 * np.pi * 0.1 * t)[None, :]
net.teach(input_data=sine, teacher=sine)

result = net.run(input_data=sine)

# Plot the output activity against the teacher
plt.figure(3)
plt.plot(sine[0])
plt.plot(result["z"][0])
```

![Sine input example](/images/sine_teacher_example.png?raw=true "Output vs teacher")

## TODO

* Add better plotting functions

* Add FORCE learning algorithm from Sussillo, Abbott 2009
