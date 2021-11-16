import gym
import gym_minigrid


def make_env(env_key, obs_type, seed=None, orig=False):
    env = gym.make(env_key)
    if orig:
        env.seed(seed)
        return env

    if obs_type == "full":
        env = gym_minigrid.wrappers.FullyObsWrapper(env)

    return env
