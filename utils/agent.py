import torch

import utils
from .other import device
from algos.model import ACModel


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir,
                 argmax=False, num_envs=1, use_memory=False, use_text=False):
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)

        self.argmax = argmax
        self.num_envs = num_envs

        models = utils.get_model_state(model_dir)
        self.acmodels = []

        for model in models:
            acmodel = ACModel(obs_space, action_space, use_memory=use_memory)

            acmodel.load_state_dict(model)
            acmodel.to(device)
            acmodel.eval()
            self.acmodels.append(acmodel)

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=device)

        actions = []
        for aid in range(len(self.acmodels)):
            acmodel = self.acmodels[aid]
            with torch.no_grad():
                dist, _ = acmodel(preprocessed_obss[aid])

            if self.argmax:
                action = dist.probs.max(1, keepdim=True)[1]
            else:
                action = dist.sample()

            actions.append(action)

        return actions

    def get_action(self, obs):
        return self.get_actions([obs])

