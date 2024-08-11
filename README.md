# tum-adlr-15
Advanced Deep Learning for Robotics - SS24 - Project 12: Trajectory Planning with Moving Obstacles

## Getting Started

All experiments were conducted using Python 3.7.16 and package versions as specified in the ```pyproject.toml``` file.
Assuming you have your preferred virtual environment set up, the environments in this repository have to be registered and all dependencies have to be installed:

```bash
cd tum-adlr-15/
pip install .
```

## End-to-End Reinforcement Learning

Training your own agent is easily possible, although you might have to tweak environment settings and reward function to your own liking using your favorite code editor.  
Then use the following command:

```bash
python train_agent.py <option> <arg>
```

Available options:
- ``-s`` / ``--start``: Start training a new agent for ``<arg>`` timesteps
- ``-r`` / ``--resume``: Continue training a pretrained agent for ``<arg>`` timesteps
- ``-e`` / ``--eval``: Evaluate the performance of a trained agent over ``<arg>`` timesteps

Fine-tuning the environment before (or after) training is possible and explained [here](https://github.com/FelixWaiblinger/tum-adlr-15/tree/main/adlr_environments#readme).

## State representation

There's also the option to tune your own neural network for state representation using:

```bash
python train_nn.py <option> <arg>
```

Available options:
- ``-r`` / ``--record``: Record ``<arg>`` image data samples from environment resets
- ``-t`` / ``--train``: Train an autoencoder for ``<arg>`` epochs on recorded data
- ``-e`` / ``--eval``: Evaluate the performance of an autoencoder (``image`` / ``loss``)

## Playtime

And finally, a mode to evaluate your motion planning abilities!  
Using commandline arguments you can choose how to play:

```bash
python play.py --level <level> --input <input> [--uncertainty]
```

Available options:
- ``-l`` / ``--level``: one of [ ```1```, ```2```, ```3``` ]
- ``-i`` / ``input``: one of [ ```mouse```, ```keyboard```, ```controller```, ```agent``` ]
- ``-u`` / ``--uncertainty``: Choose to play with noisy observations

**NOTE**: using ```--input agent``` lets you compare your times to our latest benchmark agent.

## Important References & Links

- Basis Point Sets Repository: https://github.com/sergeyprokudin/bps
