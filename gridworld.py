import random
import time
import sys
import numpy as np
import warnings

from enum import Enum

try:
    import pygame
except ImportError:
    warnings.warn('PyGame not detected, trying to run without it.')

Direction = Enum('Direction', 'left right up down')


class GridWorld:
    PENALTY = 0.01

    def __init__(self, grid=None, width=8, height=8, entities=None):
        if grid:
            self.grid = grid
        else:
            self.grid = GridWorld.generate(entities, width, height)

        self.width = len(self.grid)
        self.height = len(self.grid[0])

        empty_indices = []

        for x in range(self.width):
            for y in range(self.height):
                if self.grid[x][y] == Empty:
                    empty_indices.append((x, y))

                self.grid[x][y] = self.grid[x][y](self, x, y)

        x, y = random.choice(empty_indices)
        self.agent = Agent(self, x, y)
        self.history = [self.current_state()]
        self._total_reward = 0
        self._t = 0

    def state(self, memory=1):
        result = []
        missing_length = np.max([0, memory - len(self.history)])

        for _ in range(missing_length):
            result.append(self.history[0])

        result += self.history[-memory:]

        return np.reshape(result, [-1, memory, self.width, self.height]).transpose(0, 2, 3, 1)

    def current_state(self):
        return [[self.id(x, y) for y in range(self.height)] for x in range(self.width)]

    def id(self, x, y):
        return self.agent.id() if self.agent.x == x and self.agent.y == y else self.grid[x][y].id()

    def terminal(self):
        x, y = self.agent.x, self.agent.y

        return self.grid[x][y].terminal()

    def total_reward(self):
        return self._total_reward

    def t(self):
        return self._t

    def color(self, x, y):
        if self.agent.x == x and self.agent.y == y:
            return self.agent.color()
        else:
            return self.grid[x][y].color()

    def surface(self, x, y, size):
        if self.agent.x == x and self.agent.y == y:
            return self.agent.surface(size)
        else:
            return self.grid[x][y].surface(size)

    def act(self, action):
        return self.move(Direction(action + 1))

    def move(self, direction):
        self.agent.move(direction)

        reward = self.grid[self.agent.x][self.agent.y].reward() - GridWorld.PENALTY

        self.history.append(self.current_state())

        self._total_reward += reward
        self._t += 1

        return reward

    @staticmethod
    def generate(entities, width, height):
        grid = [[None for _ in range(height)] for _ in range(width)]

        for x in range(width):
            grid[x][0] = Wall
            grid[x][-1] = Wall

        for y in range(height):
            grid[0][y] = Wall
            grid[-1][y] = Wall

        count = 0

        for key, value in entities.iteritems():
            if key is not Empty:
                count += value

        empty_spaces = (width - 2) * (height - 2)

        assert count < empty_spaces

        entities[Empty] = empty_spaces - count

        indices = [(x, y) for x in range(1, width - 1) for y in range(1, height - 1)]
        random.shuffle(indices)
        index = 0

        for key, value in entities.iteritems():
            for _ in range(value):
                x, y = indices[index]
                index += 1

                grid[x][y] = key

        return grid


class CounterMetaClass(type):
    counter = 0

    def __new__(mcs, name, bases, attributes):
        attributes['ID'] = CounterMetaClass.counter
        CounterMetaClass.counter += 1

        return type.__new__(mcs, name, bases, attributes)


class Entity:
    __metaclass__ = CounterMetaClass

    COLOR = '#000000'
    IMG = None
    SURFACE = None

    def __init__(self, grid_world, x, y):
        self.grid_world = grid_world
        self.x = x
        self.y = y

    def id(self):
        return self.__class__.ID

    def color(self):
        return self.__class__.COLOR

    def surface(self, size):
        return self.__class__._surface(size)

    @classmethod
    def _surface(cls, size):
        if cls.IMG:
            if not cls.SURFACE:
                cls.SURFACE = pygame.transform.scale(pygame.image.load(cls.IMG), (size, size))

            return cls.SURFACE
        else:
            return None


class Agent(Entity):
    COLOR = '#B873FF'
    IMG = 'img/agent.png'

    def move(self, direction):
        x, y = self.x, self.y

        if direction == Direction.up and y > 0:
            y -= 1
        if direction == Direction.down and y < self.grid_world.height - 1:
            y += 1
        if direction == Direction.left and x > 0:
            x -= 1
        if direction == Direction.right and x < self.grid_world.width - 1:
            x += 1

        return self.grid_world.grid[x][y].interact(self)


class Block(Entity):
    REWARD = 0
    TERMINAL = False

    def __init__(self, grid, x, y):
        super(Block, self).__init__(grid, x, y)

    def reward(self):
        return self.__class__.REWARD

    def terminal(self):
        return self.__class__.TERMINAL

    def interact(self, agent):
        agent.x = self.x
        agent.y = self.y


class Empty(Block):
    COLOR = '#D4D1CA'
    IMG = 'img/ground.jpg'


class Goal(Block):
    REWARD = 1
    TERMINAL = True
    COLOR = '#49732C'
    IMG = 'img/goal.png'


class Coin(Block):
    REWARD = 0.1
    COLOR = '#FFD225'
    IMG = 'img/coin.png'

    def __init__(self, grid, x, y):
        super(Coin, self).__init__(grid, x, y)

        self.active = True

    def id(self):
        return self.__class__.ID if self.active else Empty.ID

    def color(self):
        return self.__class__.COLOR if self.active else Empty.COLOR

    def surface(self, size):
        return self.__class__._surface(size) if self.active else Empty._surface(size)

    def reward(self):
        return self.__class__.REWARD if self.active else Empty.REWARD

    def interact(self, agent):
        super(Coin, self).interact(agent)

        self.active = False


class Water(Block):
    REWARD = -0.1
    COLOR = '#0A73FF'
    IMG = 'img/water.png'


class Fire(Block):
    REWARD = -1
    TERMINAL = True
    COLOR = '#B20702'
    IMG = 'img/fire.png'


class Wall(Block):
    COLOR = '#422C25'
    IMG = 'img/wall.png'

    def interact(self, agent):
        pass


class Portal(Block):
    COLOR = '#481B5E'
    IMG = 'img/portal.png'

    def interact(self, agent):
        portals = []

        for x in range(self.grid_world.width):
            for y in range(self.grid_world.height):
                if isinstance(self.grid_world.grid[x][y], Portal) and (x, y) != (self.x, self.y):
                    portals.append((x, y))

        if len(portals) > 0:
            x, y = random.choice(portals)
        else:
            x, y = self.x, self.y

        agent.x = x
        agent.y = y


class Switch(Block):
    COLOR = '#218B87'
    IMG = 'img/switch.png'

    def __init__(self, grid, x, y):
        super(Switch, self).__init__(grid, x, y)

        self.active = True

    def id(self):
        return self.__class__.ID if self.active else Empty.ID

    def color(self):
        return self.__class__.COLOR if self.active else Empty.COLOR

    def surface(self, size):
        return self.__class__._surface(size) if self.active else Empty._surface(size)

    def interact(self, agent):
        if self.active:
            for x in range(self.grid_world.width):
                for y in range(self.grid_world.height):
                    if isinstance(self.grid_world.grid[x][y], Switch):
                        self.grid_world.grid[x][y].active = False

                    if isinstance(self.grid_world.grid[x][y], Door):
                        self.grid_world.grid[x][y].open = True

        agent.x = self.x
        agent.y = self.y


class Door(Block):
    COLOR = '#B63F00'
    IMG = 'img/door.png'

    def __init__(self, grid, x, y):
        super(Door, self).__init__(grid, x, y)

        self.open = False

    def id(self):
        return self.__class__.ID if not self.open else Empty.ID

    def color(self):
        return self.__class__.COLOR if not self.open else Empty.COLOR

    def surface(self, size):
        return self.__class__._surface(size) if not self.open else Empty._surface(size)

    def interact(self, agent):
        if self.open:
            agent.x = self.x
            agent.y = self.y


class Display:
    def __init__(self, width=8, height=8, field_size=32):
        pygame.init()
        pygame.display.set_caption('GridWorld')

        self.width = width
        self.height = height
        self.field_size = field_size
        self.screen = pygame.display.set_mode((self.width * field_size, self.height * field_size))

    def xy2rect(self, x, y, padding=1):
        return pygame.Rect(x * self.field_size + padding, y * self.field_size + padding,
                           self.field_size - 2 * padding, self.field_size - 2 * padding)

    def draw(self, grid_world, rewards=None):
        self.screen.fill((255, 255, 255))

        for x in range(self.width):
            for y in range(self.height):
                if grid_world.grid[x][y] is not Empty:
                    self.screen.blit(Empty._surface(self.field_size), (x * self.field_size, y * self.field_size))

                surface = grid_world.surface(x, y, self.field_size)

                if surface:
                    self.screen.blit(surface, (x * self.field_size, y * self.field_size))
                else:
                    pygame.draw.rect(self.screen, pygame.Color(grid_world.color(x, y)), self.xy2rect(x, y))

        if rewards is not None:
            for action in range(len(rewards)):
                x, y = (grid_world.agent.x, grid_world.agent.y)

                direction = Direction(action + 1)

                if direction == Direction.up:
                    y -= 1
                if direction == Direction.down:
                    y += 1
                if direction == Direction.left:
                    x -= 1
                if direction == Direction.right:
                    x += 1

                rect = self.xy2rect(x, y)
                font = pygame.font.Font(None, 18)

                if action == np.argmax(rewards):
                    color = '#8C151B'
                else:
                    color = '#DADAD5'

                text = font.render('%.2f' % rewards[action], 1, pygame.Color(color))
                text_position = text.get_rect()
                text_position.center = rect.center

                self.screen.blit(text, text_position)

        pygame.display.update()


if __name__ == '__main__':
    gw = GridWorld(entities={Goal: 1})
    display = Display()

    action_mappings = {
        pygame.K_LEFT: Direction.left, pygame.K_UP: Direction.up,
        pygame.K_RIGHT: Direction.right, pygame.K_DOWN: Direction.down
    }

    while not gw.terminal():
        display.draw(gw)

        direction = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                try:
                    direction = action_mappings[event.key]
                except KeyError:
                    continue

        if direction:
            gw.move(direction)

        time.sleep(0.05)
