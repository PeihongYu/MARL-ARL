from abc import ABC, abstractmethod
import torch
import numpy as np

from utils import DictList, ParallelEnv

class BaseAlgo(ABC):
    """The base class for RL algorithms."""

    def __init__(self, envs, acmodels, priors, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                 value_loss_coef, max_grad_norm, recurrence, preprocess_obss, share_reward):
        """
        Initializes a `BaseAlgo` instance.

        Parameters:
        ----------
        envs : list
            a list of environments that will be run in parallel
        acmodels : torch.Module
            the model
        priors : list
            a list of prior state-action occupancy measure
        num_frames_per_proc : int
            the number of frames collected by every process for an update
        discount : float
            the discount for future rewards
        lr : float
            the learning rate for optimizers
        gae_lambda : float
            the lambda coefficient in the GAE formula
            ([Schulman et al., 2015](https://arxiv.org/abs/1506.02438))
        entropy_coef : float
            the weight of the entropy cost in the final objective
        value_loss_coef : float
            the weight of the value loss in the final objective
        max_grad_norm : float
            gradient will be clipped to be at most this value
        recurrence : int
            the number of steps the gradient is propagated back in time
        preprocess_obss : function
            a function that takes observations returned by the environment
            and converts them into the format that the model can handle
        reshape_reward : function
            a function that shapes the reward, takes an
            (observation, action, reward, done) tuple as an input
        """

        # Store parameters
        self.share_reward = share_reward
        self.agent_num = len(envs[0].agents)
        self.env = ParallelEnv(envs)
        self.acmodels = acmodels
        self.priors = priors
        self.device = device
        self.num_frames_per_proc = num_frames_per_proc
        self.discount = discount
        self.lr = lr
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.recurrence = recurrence
        self.preprocess_obss = preprocess_obss

        # Control parameters

        assert self.recurrence == 1
        assert self.num_frames_per_proc % self.recurrence == 0

        # Configure acmodels

        for acmodel in self.acmodels:
            acmodel.to(self.device)
            acmodel.train()

        # Store helpers values

        self.num_procs = len(envs)
        self.num_frames = self.num_frames_per_proc * self.num_procs

        # Initialize experience values

        shape = (self.num_frames_per_proc, self.num_procs, self.agent_num)

        self.obs = self.env.reset()
        self.obss = [None]*(shape[0])
        self.mask = torch.ones(shape[1], device=self.device)
        self.masks = torch.zeros(shape[0:2], device=self.device)
        self.actions = torch.zeros(*shape, device=self.device, dtype=torch.int)
        self.values = torch.zeros(*shape, device=self.device)
        self.rewards = torch.zeros(*shape, device=self.device)
        self.advantages = torch.zeros(*shape, device=self.device)
        self.log_probs = torch.zeros(*shape, device=self.device)

        # Initialize log values

        self.log_episode_return = torch.zeros(shape[1:], device=self.device)
        self.log_episode_num_frames = torch.zeros(self.num_procs, device=self.device)

        self.log_done_counter = 0
        self.log_return = [[0] * self.agent_num] * self.num_procs
        self.log_num_frames = [0] * self.num_procs

    def calculate_lambda(self, N=100):

        lambdas = [{}] * self.agent_num
        shape = (self.num_procs, self.agent_num)
        actions = torch.zeros(*shape, device=self.device)

        for i in range(N):
            done = [False]
            obs = self.env.reset()
            count = 0
            local_lambdas = [{}] * self.agent_num

            while not done[0]:
                preprocessed_obs = self.preprocess_obss(obs, device=self.device)
                with torch.no_grad():
                    for aid in range(self.agent_num):
                        dist, value = self.acmodels[aid](preprocessed_obs[aid])
                        action = dist.sample()
                        actions[:, aid] = action

                obs, reward, done, info = self.env.step(actions.cpu().numpy())
                count += 1
                for pid in range(self.num_procs):
                    for aid in range(self.agent_num):
                        key = obs[pid][aid]['image'].tobytes()

                        if local_lambdas[aid].get(key):
                            local_lambdas[aid][key] += 1
                        else:
                            local_lambdas[aid][key] = 1

            for aid in range(self.agent_num):
                for key in local_lambdas[aid].keys():
                    if lambdas[aid].get(key):
                        lambdas[aid][key] += local_lambdas[aid][key] / count
                    else:
                        lambdas[aid][key] = local_lambdas[aid][key] / count

        for aid in range(self.agent_num):
            for key in lambdas[aid].keys():
                lambdas[aid][key] /= N

        self.lambdas = lambdas

        return lambdas

    def collect_experiences(self):
        """Collects rollouts and computes advantages.

        Runs several environments concurrently. The next actions are computed
        in a batch mode for all environments at the same time. The rollouts
        and advantages from all environments are concatenated together.

        Returns
        -------
        exps : DictList
            Contains actions, rewards, advantages etc as attributes.
            Each attribute, e.g. `exps.reward` has a shape
            (self.num_frames_per_proc * num_envs, ...). k-th block
            of consecutive `self.num_frames_per_proc` frames contains
            data obtained from the k-th environment. Be careful not to mix
            data from different environments!
        logs : dict
            Useful stats about the training process, including the average
            reward, policy loss, value loss, etc.
        """

        if self.priors:
            self.calculate_lambda()

        shape = (self.num_procs, self.agent_num)
        actions = torch.zeros(*shape, device=self.device)
        values = torch.zeros(*shape, device=self.device)
        log_probs = torch.zeros(*shape, device=self.device)
        next_values = torch.zeros(*shape, device=self.device)

        for i in range(self.num_frames_per_proc):
            # Do one agent-environment interaction

            preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
            with torch.no_grad():
                for aid in range(self.agent_num):
                    dist, value = self.acmodels[aid](preprocessed_obs[aid])
                    action = dist.sample()
                    actions[:, aid] = action
                    values[:, aid] = (value)
                    log_probs[:, aid] = dist.log_prob(action)

            obs, reward, done, _ = self.env.step(actions.cpu().numpy())
            # obs: a tuple containing 16 observations, each observation is a list of two

            if self.priors:
                # assume priors is given by lambda[(s,a)]
                # todo: full observation state is not the same as single agent state in 2-agent env
                for pid in range(self.num_procs):
                    for aid in range(self.agent_num):
                        key = obs[pid][aid]['image'].tobytes()  # todo: change it to the marginalized obs for 2-agent env
                        prob1 = self.lambdas[aid].get(key, 0) * np.exp(log_probs[pid][aid].item())
                        prob1 = prob1 if prob1 > 0 else 1e-6

                        key = (key, actions[pid][aid].item())
                        prob2 = self.priors[aid].get(key, 0)
                        prob2 = prob2 if prob2 > 0 else 1e-6

                        reward[pid][aid] += 0.01 * (np.log(2 * prob1) - np.log(prob1 + prob2))

            # Update experiences values

            self.obss[i] = self.obs
            self.obs = obs
            self.masks[i] = self.mask
            self.mask = 1 - torch.tensor(done, device=self.device, dtype=torch.float)
            self.actions[i] = actions
            self.values[i] = values
            reward = torch.tensor(reward, device=self.device, dtype=torch.float)
            self.rewards[i] = reward
            self.log_probs[i] = log_probs

            # Update log values

            self.log_episode_return += self.rewards[i]
            self.log_episode_num_frames += torch.ones(self.num_procs, device=self.device)

            for j, done_ in enumerate(done):
                if done_:
                    self.log_done_counter += 1
                    self.log_return.append(list(self.log_episode_return[j].cpu().numpy()))
                    self.log_num_frames.append(self.log_episode_num_frames[j].item())

            self.log_episode_return *= self.mask.unsqueeze(1)
            self.log_episode_num_frames *= self.mask

        # Add advantage and return to experiences

        preprocessed_obs = self.preprocess_obss(self.obs, device=self.device)
        with torch.no_grad():
            for aid in range(self.agent_num):
                _, next_values[:, aid] = self.acmodels[aid](preprocessed_obs[aid])

        for i in reversed(range(self.num_frames_per_proc)):
            next_mask = self.masks[i+1] if i < self.num_frames_per_proc - 1 else self.mask
            next_value = self.values[i+1] if i < self.num_frames_per_proc - 1 else next_values
            next_advantage = self.advantages[i+1] if i < self.num_frames_per_proc - 1 else 0

            if self.share_reward:
                reward_mean = self.rewards[i].mean(1)
                cur_rewards = torch.vstack(tuple([reward_mean for i in range(1)])).transpose(0,1)
            else:
                cur_rewards = self.rewards[i]

            delta = cur_rewards + self.discount * next_value * next_mask.unsqueeze(1) - self.values[i]
            self.advantages[i] = delta + self.discount * self.gae_lambda * next_advantage * next_mask.unsqueeze(1)

        # Define experiences:
        #   the whole experience is the concatenation of the experience
        #   of each process.
        # In comments below:
        #   - A is self.agent_num,
        #   - T is self.num_frames_per_proc,
        #   - P is self.num_procs,
        #   - D is the dimensionality.

        exps = DictList()
        exps.obs = [self.obss[i][j]
                    for j in range(self.num_procs)
                    for i in range(self.num_frames_per_proc)]
        # for all tensors below, T x P x A -> A x P x T -> A x (P * T)
        shape = (self.agent_num, self.num_frames)
        exps.action = self.actions.transpose(0, 2).reshape(shape)
        exps.value = self.values.transpose(0, 2).reshape(shape)
        exps.reward = self.rewards.transpose(0, 2).reshape(shape)
        exps.advantage = self.advantages.transpose(0, 2).reshape(shape)
        exps.returnn = exps.value + exps.advantage
        exps.log_prob = self.log_probs.transpose(0, 2).reshape(shape)

        # Preprocess experiences

        exps.obs = self.preprocess_obss(exps.obs, device=self.device)

        # Log some values

        keep = max(self.log_done_counter, self.num_procs)

        logs = {
            "return_per_episode": self.log_return[-keep:],
            "num_frames_per_episode": self.log_num_frames[-keep:],
            "num_frames": self.num_frames
        }

        self.log_done_counter = 0
        self.log_return = self.log_return[-self.num_procs:]
        self.log_num_frames = self.log_num_frames[-self.num_procs:]

        return exps, logs

    @abstractmethod
    def update_parameters(self, exps):
        pass
