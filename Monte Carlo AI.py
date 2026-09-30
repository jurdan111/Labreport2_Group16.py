# 2048 AI using Monte Carlo search with a fixed time budget per move

#
# DEPTH sets how far each simulation plays:
#   DEPTH = None -> play until game over
#   DEPTH = 10   -> initial move + 10 random moves
#
# Press q / ESC to quit

from Game2048 import Game2048
import numpy as np
import pygame
import time

# Settings
TIME_PER_MOVE = 0.05    # seconds of thinking per move
DEPTH = 10              # None = simulate until game over

env = Game2048()
env.reset()
actions = ['left', 'right', 'up', 'down']
exit_program = False
done = False

total_time = 0          # time spent thinking in the whole game
total_sims = 0          # simulations made in the whole game
n_moves = 0             # moves played in the game

while not exit_program:
    env.render()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                exit_program = True

    if done:            # game over: keep showing the board until we quit
        continue

    # Monte Carlo search
    start_time = time.perf_counter()

    # Only use moves that actually change the board. Otherwise an illegal move
    # can win a tie at the very end of a game, and the game never ends.
    legal_actions = []
    for init_action in actions:
        sim = Game2048((env.board, env.score))
        sim.step(init_action)
        if not np.array_equal(sim.board, env.board):
            legal_actions.append(init_action)

    scores = {}
    for init_action in legal_actions:
        scores[init_action] = 0

    # One simulation per initial move at a time, until we run out of time
    while time.perf_counter() - start_time < TIME_PER_MOVE:
        for init_action in legal_actions:
            sim = Game2048((env.board, env.score))
            action = init_action
            sim_done = False
            sim_moves = 0
            while not sim_done and (DEPTH is None or sim_moves <= DEPTH):
                (board, score), reward, sim_done = sim.step(action)
                action = actions[np.random.randint(4)]
                sim_moves += 1
            scores[init_action] += score
            total_sims += 1

    # Pick the initial move with the highest total score
    action = max(scores, key=lambda key: scores[key])
    total_time += time.perf_counter() - start_time
    n_moves += 1

    (board, score), reward, done = env.step(action)

    if done:
        print(f'Game over! Score: {score}')
        print(f'Average time per move: {total_time / n_moves:.4f} s')
        print(f'Average simulations per move: {total_sims / n_moves:.0f}')

env.close()
