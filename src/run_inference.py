# from gymnasium.spaces import Box, Discrete
import torch
from gymnasium.envs.registration import register

import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# from IPython import display

from run_learning import PPOac


if __name__ == '__main__':
    torch.manual_seed(123) 
    torch.cuda.manual_seed(123) 
    np.random.seed(123) 
    # random.seed(123) 
    # torch.backends.cudnn.enabled=False 
    # torch.backends.cudnn.deterministic=True
    STEPS_MAX = 10000//2
    path_weights = 'hex_best_120.weights'
        
    register(
        id="Hexapod-v0",
        entry_point="gym_env:HexapodEnv",
        max_episode_steps=STEPS_MAX,
        reward_threshold=1.76*300,
    )
    env = gym.make('Hexapod-v0', render_mode='rgb_array', max_geom=7000)

    #Later to restore:
    n_hidden = 70
    n_hid_layers = 2
    new_agent = PPOac(env.observation_space, env.action_space, n_hidden, n_hid_layers=n_hid_layers)
    new_agent.load_state_dict(torch.load(path_weights, weights_only=True))
    new_agent.eval()


    # Execution parameters
    SHOW_ANIMATION = False
    EPISODES_MAX = 20

    # Loggers
    log_steps_number = np.zeros(EPISODES_MAX)
    log_fwd_return = np.zeros(EPISODES_MAX)

    # PQ
    for i_episode in range(EPISODES_MAX):
        observation = env.reset()[0]
        state = observation
        episode_total_return = 0
        episode_cost = 0
        episode_fwd_return = 0

        # show results
        if (i_episode + 1) % 20 == 0:
            # plt.figure(1)
            # plt.clf()
            # plt.plot([0,i_episode], [495, 495], label="threshold")
            # plt.plot(range(0,i_episode), log_steps_number[0:i_episode], label="solution 1")
            # plt.xlabel('episode')
            # plt.ylabel('episode steps')
            # plt.legend()
            # display.clear_output(wait=True)
            # plt.show()
            pass

        for t in range(STEPS_MAX):
            action, _, _ = new_agent.get_action(state)
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = observation
            episode_total_return += reward
            episode_fwd_return += info["reward_forward"]
            episode_cost += info["cost"]

            if done:
                log_steps_number[i_episode] = episode_total_return
                log_fwd_return[i_episode] = episode_fwd_return
                break
        print(f'episode {i_episode}, return_fwd: {episode_fwd_return:.2f}, ypos: {info["ypos"]:.2f}, ctrl_cost: {episode_cost}')

    print("done")

    plt.figure()
    # plt.clf()
    plt.plot([0,i_episode], [env.unwrapped.spec.reward_threshold, env.unwrapped.spec.reward_threshold], label="baseline (forward)")
    plt.plot(range(0,i_episode), log_fwd_return[0:i_episode], label="solution")
    plt.xlabel('episode')
    plt.ylabel('episode return (forward)')
    plt.legend()
    # display.clear_output(wait=True)
    plt.show()