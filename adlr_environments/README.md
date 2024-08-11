# Experimental environments

## Experiments in 2D

The environment "World2D-v0" consists of a limited, square area and includes an agent (blue), a target (red), 0...N static obstacles (black) and 0...M dynamic obstacles (green).

## Configuration

The environment setup can be configured by passing a dictionary to the ```create_env``` function.  
Available entries are:
- ```world```: A predefined world setup without randomization of positions
- ```episode_length```: Number of timesteps per episode before truncation
- ```step_length```: Maximum agent speed (relative to a world size of 2x2)
- ```size_agent```: Radius of the agent circle / collision area
- ```size_target```: Radius of the target circle / collision area
- ```num_static_obstacles```: Number of static obstacles in the environment
- ```size_static```: Radius of static obstacle circles / collision areas
- ```num_dynamic_obstacles```: Number of dynamic obstacles in the environment
- ```size_dynamic```: Radius of dynamic obstacle circles / collision areas
- ```min_speed```: Minimum dynamic obstacle speed (for random sampling)
- ```max_speed```: Maximum dynamic obstacle speed (for random sampling)
- ```uncertainty```: Flag to enable uncertainty (noise in position and image observations)
