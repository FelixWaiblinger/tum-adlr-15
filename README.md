# tum-adlr-15
Advanced Deep Learning for Robotics - SS24 - Project 12: Trajectory Planning with Moving Obstacles

## Getting Started

Assuming you have your preferred virtual environment set up, the environments in this repository have to be registered:

```bash
pip install -e adlr_environments/
```

## Environment check-up

Then simply run the simulation and witness astonishing random movements:

```bash
python simulation.py
```

## End-to-End Reinforcement Learning

Training your own agent is easily possible, although you might have to tweak environment settings and reward function to your own liking using your favorite code editor.  
Then use the following command:

```bash
python train_agent.py
```

## State representation

There's also the option to tune your own neural network for state representation using:

```bash
python train_nn.py
```

## Playtime

And finally, there's a competition... of course!
Using commandline arguments you can choose how to play:

```bash
python play.py --level <level> --input <input>
```

Available commands include:
- level: [ ```1```, ```2```, ```3``` ]
- input: [ ```mouse```, ```keyboard```, ```controller```, ```agent``` ]

**NOTE**: using ```--input agent``` lets you compare your times to our latest benchmark agent.

## Important References & Links
- Basis Point Sets Repository: https://github.com/sergeyprokudin/bps
