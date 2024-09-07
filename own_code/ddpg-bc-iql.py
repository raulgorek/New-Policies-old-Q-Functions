import argparse
import random

import os
import glob
from typing import List, Optional, Tuple, Union
import gym
import d4rl

import gym.envs
import numpy as np
import torch
import matplotlib.pyplot as plt

from copy import copy
from tqdm import tqdm
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, DiagGaussian
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.policy import CQLPolicy, IQLPolicy
from offlinerlkit.utils.logger import Logger, make_log_dirs


"""
suggested hypers
cql-weight=5.0, temperature=1.0 for all D4RL-Gym tasks
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-path", type=str, default=None, required=True)
    parser.add_argument("--task", type=str, default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--algo-name", type=str, default="DDPG+BC_IQL")

    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test", type=bool, default=False)

    return parser.parse_args()

class MLP(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim: Optional[int] = None,
        activation: torch.nn.Module = torch.nn.ReLU,
        output_activation: torch.nn.Module = torch.nn.Tanh,
        dropout_rate: Optional[float] = None,
        use_layer_norm: bool = False
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [torch.nn.Linear(in_dim, out_dim), activation()]
            if use_layer_norm:
                model += [torch.nn.LayerNorm(out_dim)]
            if dropout_rate is not None:
                model += [torch.nn.Dropout(p=dropout_rate)]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [torch.nn.Linear(hidden_dims[-1], output_dim)]
            model += [output_activation()]
            self.output_dim = output_dim
        self.model = torch.nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def create_iql(device, env: gym.Env):
    # create policy model
    hidden_dims = [256, 256]
    obs_shape = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)
    max_action = env.action_space.high[0]
    actor_lr = 3e-4
    critic_q_lr = 3e-4
    critic_v_lr = 3e-4
    tau = 0.005
    gamma = 0.99
    expectile = 0.7
    temperature = 3.0


    actor_backbone = MLP(input_dim=np.prod(obs_shape), hidden_dims=hidden_dims)
    critic_q1_backbone = MLP(input_dim=np.prod(obs_shape)+action_dim, hidden_dims=hidden_dims)
    critic_q2_backbone = MLP(input_dim=np.prod(obs_shape)+action_dim, hidden_dims=hidden_dims)
    critic_v_backbone = MLP(input_dim=np.prod(obs_shape), hidden_dims=hidden_dims)
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=action_dim,
        unbounded=False,
        conditioned_sigma=False,
        max_mu=env.action_space.high[0]
    )
    actor = ActorProb(actor_backbone, dist, device)
    critic_q1 = Critic(critic_q1_backbone, device)
    critic_q2 = Critic(critic_q2_backbone, device)
    critic_v = Critic(critic_v_backbone, device)
    
    for m in list(actor.modules()) + list(critic_q1.modules()) + list(critic_q2.modules()) + list(critic_v.modules()):
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)

    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_q1_optim = torch.optim.Adam(critic_q1.parameters(), lr=critic_q_lr)
    critic_q2_optim = torch.optim.Adam(critic_q2.parameters(), lr=critic_q_lr)
    critic_v_optim = torch.optim.Adam(critic_v.parameters(), lr=critic_v_lr)
    
    # create IQL policy
    policy = IQLPolicy(
        actor,
        critic_q1,
        critic_q2,
        critic_v,
        actor_optim,
        critic_q1_optim,
        critic_q2_optim,
        critic_v_optim,
        action_space=env.action_space,
        tau=tau,
        gamma=gamma,
        expectile=expectile,
        temperature=temperature
    )
    return policy


def evaluate_policy(policy, env, num_episodes=10):
    with torch.no_grad():
        policy.eval()
        obs: np.ndarray = env.reset()
        eval_ep_info_buffer = []
        n_episodes = 0
        episode_reward, episode_length = 0, 0

        while n_episodes < num_episodes:
            action = policy(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)).cpu().numpy()
            action = np.clip(action, env.action_space.low[0], env.action_space.high[0])
            next_obs, reward, terminal, _ = env.step(action.flatten())
            episode_reward += reward
            episode_length += 1

            obs = next_obs

            if terminal:
                eval_ep_info_buffer.append(
                    {"episode_reward": episode_reward, "episode_length": episode_length}
                )
                n_episodes += 1
                episode_reward, episode_length = 0, 0
                obs = env.reset()
        return {
            "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
            "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer]
        }


def train(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    dataset = qlearning_dataset(env)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]
    os.makedirs(f"models/alpha{args.alpha}", exist_ok=True)
    os.makedirs(f"plots/alpha{args.alpha}", exist_ok=True)
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    if args.test:
        args.epoch = 10
        log_dirs = make_log_dirs(args.task, f"{args.algo_name}_test", args.seed, vars(args))
    else:
        log_dirs = make_log_dirs(args.task, f"{args.algo_name}_alpha_{args.alpha}", args.seed, vars(args))
    
    output_config = {
        "consoleout_backup": "stdout",
        "policy_training_progress": "csv",
        "tb": "tensorboard"
    }
    logger = Logger(log_dirs, output_config)
    logger.log_hyperparameters(vars(args))

    # Create IQL Model
    iql_policy = create_iql(args.device, env)
    matching_folder = glob.glob(os.path.join(args.load_path, f"{args.task}/iql/seed_{args.seed}*"))[0]
    print(matching_folder)
    iql_dict = torch.load(f"{matching_folder}/model/policy.pth", map_location=args.device)
    iql_policy.load_state_dict(iql_dict)


    # Build new actor
    obs_shape = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)
    hidden_dims = [256, 256]
    actor_lr = 3e-4
    a_max = env.action_space.high
    a_min = env.action_space.low

    new_actor = MLP(input_dim=np.prod(obs_shape), output_dim=action_dim, hidden_dims=hidden_dims, activation=torch.nn.ReLU, use_layer_norm=True, output_activation=torch.nn.Tanh)
    actor_optim = torch.optim.Adam(new_actor.parameters(), lr=actor_lr)
    
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=obs_shape,
        obs_dtype=np.float32,
        action_dim=action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset)

    norm_returns_per_epoch = []
    gradient_steps = []
    num_timestamps = 0
    alpha = args.alpha
    for e in range(1, args.epoch + 1):
        new_actor.train()
        iql_policy.eval()

        pbar = tqdm(range(args.step_per_epoch), desc=f"Epoch #{e}/{args.epoch}")
        for it in pbar:
            batch = buffer.sample(args.batch_size)
            obs = batch["observations"]
            actions = batch["actions"]
            
            actor_optim.zero_grad()
            action = new_actor(obs)
            action = torch.clamp(action, a_min[0], a_max[0])
            q1 = iql_policy.critic_q1(obs, action)
            q2 = iql_policy.critic_q2(obs, action)
            q = torch.min(q1, q2)
            actor_loss = -q.mean()
            bc_loss = torch.nn.functional.mse_loss(action, actions)
            combined_loss = actor_loss + alpha * bc_loss
            combined_loss.backward()
            actor_optim.step()
            num_timestamps += 1
            logger.logkv("train/actor_loss", actor_loss.item())
            logger.logkv("train/bc_loss", bc_loss.item())
            logger.logkv("train/combined_loss", combined_loss.item())
        eval_info = evaluate_policy(new_actor, env, num_episodes=args.eval_episodes)
        norm_returns_per_epoch.append(100 * np.mean([d4rl.get_normalized_score(args.task, eval_info['eval/episode_reward'][i]) for i in range(args.eval_episodes)]))
        gradient_steps.append(e * args.step_per_epoch)
        logger.set_timestep(num_timestamps)
        logger.logkv("eval/normalized_episode_reward", copy(norm_returns_per_epoch[-1]))
        logger.dumpkvs()

    torch.save(new_actor.state_dict(), os.path.join(logger.model_dir, "policy.pth"))
    logger.close()


if __name__ == "__main__":
    train()