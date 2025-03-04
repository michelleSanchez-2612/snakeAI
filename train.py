# train.py
import random
import numpy as np
from game import SnakeGameAI
from utils import plot_metrics
from agents import DeepQAgent, TableQAgent, SarsaQAgent, NNAgent

def select_model():
    print("Select the model to run:")
    print("1: Q-Learning")
    print("2: SARSA")
    print("3: Neural Network")
    print("4: Deep Q-Learning")
    choice = input("Enter a number: ")
    return choice

def train():
    model_choice = select_model()
    episode_scores = []   # Score per episode
    episode_lengths = []  # Number of steps per episode
    mean_scores = []      # Cumulative average score
    record_scores = []    # Highest score reached
    total_score = 0
    record = 0
    episode_count = 0

    game = SnakeGameAI()

    if model_choice == "1":
        agent = TableQAgent()
        model_name = "Q_Learning"
    elif model_choice == "2":
        agent = SarsaQAgent()
        model_name = "SARSA Learning"
    elif model_choice == "3":
        agent = NNAgent()
        model_name = "Neural Network"
    elif model_choice == "4":
        agent = DeepQAgent()
        model_name = "Deep_Q_Learning"
    else:
        print("Invalid choice. Defaulting to Deep Q-Learning.")
        agent = DeepQAgent()
        model_name = "Deep_Q_Learning"

    while True:
        state_old = agent.get_state(game)
        action = agent.get_action(state_old)
        reward, done, score = game.play_step(action)
        state_new = agent.get_state(game)
        
        # Update agent based on the model type
        if model_choice == "1": #Q-Learning
            agent.update(state_old, action, reward, state_new, done)
        elif model_choice == "2": #SARSA
            next_action = agent.get_action(state_new)
            agent.update_sarsa(state_old, action, reward, state_new, next_action, done)
        elif model_choice == "3":
            agent.store_reward(reward)
        elif model_choice == "4": #Deep Q-Learning
            agent.train_short_memory(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)
        
        if done:
            episode_count += 1
            episode_length = game.frame_iteration
            episode_lengths.append(episode_length)
            episode_scores.append(score)
            total_score += score
            mean_score = total_score / episode_count
            mean_scores.append(mean_score)
            
            if score > record:
                record = score
                # If Deep Q-Learning, save the model when a new record is reached.
                if model_choice == "4":
                    agent.model.save()
            record_scores.append(record)

            # For REINFORCE, update the policy at the end of the episode
            if model_choice == "3":
                agent.update_policy()
            
            game.reset()
            # For Deep Q-Learning, train on the replay memory
            if model_choice == "4":
                agent.train_long_memory()
            if model_choice in ["1", "2", "4"]:
                agent.update_epsilon()
            
            print(f"Game {episode_count} | Score: {score} | Record: {record} | Mean Score: {mean_score:.2f} | Length: {episode_length}")
            if episode_count % 50 == 0:
                plot_metrics(episode_scores, mean_scores, record_scores, episode_lengths, episode_count, model_name=model_name)

if __name__ == '__main__':
    train()
