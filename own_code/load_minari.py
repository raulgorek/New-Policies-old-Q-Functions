import minari
import numpy as np
import gym

def load_minari(dataset: minari.MinariDataset, env: gym.Env):
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    episode_step = 0
    for episode in dataset.iterate_episodes():
        for i in range(episode.actions.shape[0]):
            obs = episode.observations[i].astpye(np.float32)
            new_obs = episode.observations[i+1].astype(np.float32)
            action = episode.actions[i].astype(np.float32)
            reward = episode.rewards[i].astype(np.float32)
            done_bool = bool(episode.terminations[i] + episode.truncations[i])
            final_timestep = (episode_step == env._max_episode_steps - 1)
            if final_timestep:
                episode_step = 0
                continue
            if done_bool or final_timestep:
                episode_step = 0
            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)
            episode_step += 1
        
        return {
            'observations': np.array(obs_),
            'actions': np.array(action_),
            'next_observations': np.array(next_obs_),
            'rewards': np.array(reward_),
            'terminals': np.array(done_),
        }


if __name__ == "__main__":
    dataset = minari.load_dataset("hopper-medium-v2")
    load_minari(dataset)