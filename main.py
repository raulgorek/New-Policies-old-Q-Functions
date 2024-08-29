import argparse
import random

import os
import glob
import gym
import d4rl

import gym.envs
import numpy as np
import torch

from tqdm import tqdm
from offlinerlkit.nets import MLP
from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian, DiagGaussian
from offlinerlkit.utils.load_dataset import qlearning_dataset
from offlinerlkit.buffer import ReplayBuffer
from offlinerlkit.utils.logger import Logger, make_log_dirs
from offlinerlkit.policy import CQLPolicy, IQLPolicy


"""
suggested hypers
cql-weight=5.0, temperature=1.0 for all D4RL-Gym tasks
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load-path", type=str, default=None, required=True)
    parser.add_argument("--task", type=str, default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


def create_cql(device, env: gym.Env):
    # create policy model
    hidden_dims = [256, 256, 256]
    obs_shape = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)
    max_action = env.action_space.high[0]
    actor_lr = 1e-4
    critic_lr = 3e-4
    auto_alpha = True
    target_entropy = None
    alpha_lr = 1e-4
    tau = 0.005
    gamma = 0.99
    cql_weight = 5.0
    temperature = 1.0
    max_q_backup = False
    deterministic_backup = True
    with_lagrange = False
    lagrange_threshold = 10.0
    cql_alpha_lr = 3e-4
    num_repeat_actions = 10

    actor_backbone = MLP(input_dim=np.prod(obs_shape), hidden_dims=hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(obs_shape) + action_dim, hidden_dims=hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(obs_shape) + action_dim, hidden_dims=hidden_dims)
    dist = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=action_dim,
        unbounded=True,
        conditioned_sigma=True,
        max_mu=max_action
    )
    actor = ActorProb(actor_backbone, dist, device)
    critic1 = Critic(critic1_backbone, device)
    critic2 = Critic(critic2_backbone, device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

    if auto_alpha:
        target_entropy = target_entropy if target_entropy \
            else -np.prod(env.action_space.shape)

        target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = 0.2

    # create policy
    policy = CQLPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        tau=tau,
        gamma=gamma,
        alpha=alpha,
        cql_weight=cql_weight,
        temperature=temperature,
        max_q_backup=max_q_backup,
        deterministic_backup=deterministic_backup,
        with_lagrange=with_lagrange,
        lagrange_threshold=lagrange_threshold,
        cql_alpha_lr=cql_alpha_lr,
        num_repeart_actions=num_repeat_actions
    )
    return policy


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
        obs = env.reset()
        eval_ep_info_buffer = []
        n_episodes = 0
        episode_reward, episode_length = 0, 0

        while n_episodes < num_episodes:
            dist = policy(torch.tensor(obs).unsqueeze(0))
            action = dist.mode()
            if isinstance(action, tuple):
                action = action[0]
            action = action.cpu().numpy()
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


def linearly_divide_action_space(min_bounds, max_bounds, num_divisions):
    """
    Linearly divides the action space into a grid based on the min and max bounds.

    Parameters:
    - min_bounds: Array of minimum values for each action dimension.
    - max_bounds: Array of maximum values for each action dimension.
    - num_divisions: Number of divisions per dimension.

    Returns:
    - action_grid: A list of actions representing the grid.
    """
    # Create a list of linearly spaced values for each dimension
    action_grid = [
        torch.linspace(min_bounds[i], max_bounds[i], num_divisions)
        for i in range(len(min_bounds))
    ]
    
    # Create the Cartesian product of the grids to form the action grid
    action_grid = torch.stack(torch.meshgrid(*action_grid), dim=-1).reshape(-1, len(min_bounds))
    
    # Remove duplicate rows (not usually needed in torch, but keeping for consistency with numpy code)
    action_grid = torch.unique(action_grid, dim=0)
    
    return action_grid

def approx_value_function(q_function, obs, actions):
    """
    Approximates the value function using a given Q-function.

    Parameters:
    - q_function: The Q-function to use for approximating the value function.
    - obs: The observation to use for the value function.
    - actions: The actions to use for the value function.

    Returns:
    - values: The approximated value function.
    """
    with torch.no_grad():
        q = torch.cat([q_function(obs, action.expand(obs.shape[0], action.shape[1])) for action in actions.unsqueeze(1)], dim=1)
    v, _ = torch.max(q, dim=1, keepdim=True)
    return v


def train(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    dataset = qlearning_dataset(env)
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # Create CQL Model
    cql_policy = create_cql(args.device, env)
    matching_folder = glob.glob(os.path.join(args.load_path, f"{args.task}/cql/seed_{args.seed}*"))[0]
    print(matching_folder)
    cql_dict = torch.load(f"{matching_folder}/model/policy.pth", map_location=args.device)
    cql_policy.load_state_dict(cql_dict)
    # Create IQL Model
    iql_policy = create_iql(args.device, env)
    matching_folder = glob.glob(os.path.join(args.load_path, f"{args.task}/iql/seed_{args.seed}*"))[0]
    print(matching_folder)
    iql_dict = torch.load(f"{matching_folder}/model/policy.pth", map_location=args.device)
    iql_policy.load_state_dict(iql_dict)
    print("Loaded models")
    print("IQL Q-Net")
    print(iql_policy.critic_q1)
    print("CQL Q-Net")
    print(cql_policy.critic1)
    print("IQL Policy-Net")
    print(iql_policy.actor)
    print("CQL Policy-Net")
    print(cql_policy.actor)
    test_obs = env.observation_space.sample()
    print("sampled obs")
    print(test_obs)
    print("IQL Response")
    print(iql_policy.actor.forward(torch.tensor(test_obs).unsqueeze(0)))
    print("CQL Response")
    print(cql_policy.actor.forward(torch.tensor(test_obs).unsqueeze(0)))

    # Build new actor
    obs_shape = env.observation_space.shape
    action_dim = np.prod(env.action_space.shape)
    hidden_dims = [256, 256]
    temperature = 1.0
    actor_lr = 3e-4
    a_max = env.action_space.high
    a_min = env.action_space.low
    approx_v_bins = 10
    myhyper = 0.5
    action_grid = linearly_divide_action_space(a_min, a_max, approx_v_bins)
    print("Num actions to approximate V: ", len(action_grid))

    actor_backbone = MLP(input_dim=np.prod(obs_shape), hidden_dims=hidden_dims)
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=action_dim,
        unbounded=False,
        conditioned_sigma=False,
        max_mu=env.action_space.high[0]
    )
    new_actor = ActorProb(actor_backbone, dist, args.device)
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
    cql_eval = evaluate_policy(cql_policy.actor, env, num_episodes=args.eval_episodes)
    iql_eval = evaluate_policy(iql_policy.actor, env, num_episodes=args.eval_episodes)
    cql_norm_return = 100 * np.mean([d4rl.get_normalized_score(args.task, cql_eval["eval/episode_reward"][i]) for i in range(args.eval_episodes)])
    iql_norm_return = 100 * np.mean([d4rl.get_normalized_score(args.task, iql_eval["eval/episode_reward"][i]) for i in range(args.eval_episodes)])


    for e in range(1, args.epoch + 1):
        new_actor.train()
        cql_policy.eval()
        iql_policy.eval()

        pbar = tqdm(range(args.step_per_epoch), desc=f"Epoch #{e}/{args.epoch}")
        for it in pbar:
            batch = buffer.sample(args.batch_size)
            obs = batch["observations"]
            actions = batch["actions"]
            with torch.no_grad():
                q_cql1, q_cql2 = cql_policy.critic1(obs, actions), cql_policy.critic2(obs, actions)
                q_iql1, q_iql2 = iql_policy.critic_q1(obs, actions), iql_policy.critic_q2(obs, actions)
                q_cql = torch.min(q_cql1, q_cql2)
                q_iql = torch.min(q_iql1, q_iql2)
                q_use = (1 - myhyper) * q_cql + myhyper * q_iql
                v_cql1 = approx_value_function(cql_policy.critic1, obs, action_grid)
                v_cql2 = approx_value_function(cql_policy.critic2, obs, action_grid)
                v_cql = torch.min(v_cql1, v_cql2)
                v_iql = iql_policy.critic_v(obs)
                v_use = (1 - myhyper) * v_cql + myhyper * v_iql
                exp_a = torch.exp((q_use - v_use) * temperature)
                exp_a = torch.clip(exp_a, None, 100.0)
            dist = new_actor(obs)
            log_probs = dist.log_prob(actions)
            actor_loss = -(exp_a * log_probs).mean()

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()
        print(f"Epoch {e} complete")
        print(f"CQL Norm Return: {cql_norm_return}")
        print(f"IQL Norm Return: {iql_norm_return}")
        eval_info = evaluate_policy(new_actor, env, num_episodes=args.eval_episodes)
        print(f"Mixed Policy Norm Return: {100 * np.mean([d4rl.get_normalized_score(args.task, eval_info['eval/episode_reward'][i]) for i in range(args.eval_episodes)])}")
        print("Saving model")
        torch.save(new_actor.state_dict(), f"models/cql_iql_ens_{args.task}_seed_{args.seed}")


if __name__ == "__main__":
    train()