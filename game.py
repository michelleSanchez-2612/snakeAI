import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

# Initialize pygame
pygame.init()

# Set up font for score display
font = pygame.font.SysFont('arial', 25)
Point = namedtuple('Point', 'x, y')

# Define colors (RGB format)
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

# Block size and game speed
BLOCK_SIZE = 20
SPEED = 40

# Define possible directions
class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        """Initialize game parameters and set up display."""
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Reset the game state to start a new game."""
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)
        ]
        self.score = 0
        self.food = None
        self.place_food()
        self.frame_iteration = 0

    def place_food(self):
        """Place food in a random location that is not on the snake."""
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.place_food()

    def is_collision(self, pt=None):
        """Check if a collision occurs (with wall or self)."""
        if pt is None:
            pt = self.head
        # Check wall collision
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # Check self collision
        if pt in self.snake[1:]:
            return True #snake hits its body        
        return False

    def update_ui(self):
        """Update the user interface: draw snake, food, and score."""
        self.display.fill(BLACK)
        # Draw snake head in green
        head = self.snake[0]
        pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(head.x, head.y, BLOCK_SIZE, BLOCK_SIZE))
        # Draw the rest of the snake body in blue
        for pt in self.snake[1:]:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        # Draw food in red
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        # Display score
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def move(self, action):
        """
        Move the snake based on the action.
        Action is a one-hot encoded vector:
          [1, 0, 0] -> move straight,
          [0, 1, 0] -> turn right,
          [0, 0, 1] -> turn left.
        """
        # Clockwise order of directions
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # Straight
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Right turn
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Left turn

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    def play_step(self, action):
        """
        Execute one step of the game.
        Returns a tuple (reward, game_over, score).
        """
        self.frame_iteration += 1
        # Handle events (quit event)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Calculate previous Manhattan distance to food
        prev_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        # Move snake based on action
        self.move(action)
        # Calculate new Manhattan distance to food
        new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        # Reward shaping: reward for moving closer to food
        reward_shaping = (prev_distance - new_distance) * 0.1

        self.snake.insert(0, self.head)
        game_over = False

        # Check for collisions or frame limit exceeded
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score

        # Check if food is eaten
        if self.head == self.food:
            self.score += 1
            reward = 10 + reward_shaping  # bonus reward for eating food
            self.place_food()
        else:
            reward = reward_shaping
            self.snake.pop()

        self.update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score
