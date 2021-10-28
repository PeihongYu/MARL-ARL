import gym
import gym_minigrid


def make_env(env_key, seed=None, orig=False):
    env = gym.make(env_key)
    if orig:
        env.seed(seed)
        return env

    env = gym_minigrid.wrappers.FullyObsWrapper(env)

    # if len(env.agents_type) == 1:
    #     env = gym_minigrid.wrappers.SingleAgentWrapper(env)
    # else:
    #     env = gym_minigrid.wrappers.TwoAgentWrapper(env, mode="full")
    # env.seed(seed)
    return env
