import numpy
import torch
import torch.nn.functional as F

from algos.base import BaseAlgo


class A2CAlgo(BaseAlgo):

    def __init__(self, env, acmodels, priors=None, device=None, num_frames_per_proc=None, discount=0.99, lr=0.01, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 rmsprop_alpha=0.99, rmsprop_eps=1e-8, preprocess_obss=None, share_reward=False):
        num_frames_per_proc = num_frames_per_proc or 16

        super().__init__(env, acmodels, priors, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, share_reward)

        self.optimizers = [torch.optim.RMSprop(acmodel.parameters(), lr, alpha=rmsprop_alpha, eps=rmsprop_eps)
                           for acmodel in self.acmodels]

    def update_parameters(self, exps):

        # Compute starting indexes

        inds = self._get_starting_indexes()

        # Initialize update values

        all_update_entropy = []
        all_update_value = []
        all_update_policy_loss = []
        all_update_value_loss = []
        all_update_grad_norm = []

        for aid in range(self.agent_num):

            # Initialize update values for each agent

            update_entropy = 0
            update_value = 0
            update_policy_loss = 0
            update_value_loss = 0
            update_loss = 0

            for i in range(self.recurrence):
                # Create a sub-batch of experience

                sb = exps[aid][inds + i]

                # Compute loss

                dist, value = self.acmodels[aid](sb.obs)

                entropy = dist.entropy().mean()

                policy_loss = -(dist.log_prob(sb.action) * sb.advantage).mean()

                value_loss = (value - sb.returnn).pow(2).mean()

                loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                # Update batch values

                update_entropy += entropy.item()
                update_value += value.mean().item()
                update_policy_loss += policy_loss.item()
                update_value_loss += value_loss.item()
                update_loss += loss

            # Update update values

            update_entropy /= self.recurrence
            update_value /= self.recurrence
            update_policy_loss /= self.recurrence
            update_value_loss /= self.recurrence
            update_loss /= self.recurrence

            # Update actor-critic

            self.optimizers[aid].zero_grad()
            update_loss.backward()
            update_grad_norm = sum(p.grad.data.norm(2) ** 2 for p in self.acmodels[aid].parameters()) ** 0.5
            torch.nn.utils.clip_grad_norm_(self.acmodels[aid].parameters(), self.max_grad_norm)
            self.optimizers[aid].step()

            # store logs
            all_update_entropy.append(update_entropy)
            all_update_value.append(update_value)
            all_update_policy_loss.append(update_policy_loss)
            all_update_value_loss.append(update_value_loss)
            all_update_grad_norm.append(update_grad_norm)

        # Log some values

        logs = {
            "entropy": all_update_entropy,
            "value": all_update_value,
            "policy_loss": all_update_policy_loss,
            "value_loss": all_update_value_loss,
            "grad_norm": all_update_grad_norm
        }

        return logs

    def _get_starting_indexes(self):
        """Gives the indexes of the observations given to the model and the
        experiences used to compute the loss at first.

        The indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`. If the model is not recurrent, they are all the
        integers from 0 to `self.num_frames`.

        Returns
        -------
        starting_indexes : list of int
            the indexes of the experiences to be used at first
        """

        starting_indexes = numpy.arange(0, self.num_frames, self.recurrence)
        return starting_indexes

