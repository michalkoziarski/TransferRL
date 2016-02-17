import pygame
import random
import time
import sys

from enum import Enum


Direction = Enum('Direction', 'left right up down')


class GridWorld:
    PENALTY = 0.01

    def __init__(self, grid=None, width=16, height=16, entities=None):
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

        self.total_reward = 0

    def state(self):
        return [[self.id(x, y) for y in range(self.height)] for x in range(self.width)]

    def id(self, x, y):
        return self.agent.id() if self.agent.x == x and self.agent.y == y else self.grid[x][y].id()

    def terminal(self):
        x, y = self.agent.x, self.agent.y

        return self.grid[x][y].terminal()

    def color(self, x, y):
        if self.agent.x == x and self.agent.y == y:
            return self.agent.color()
        else:
            return self.grid[x][y].color()

    def move(self, direction):
        self.agent.move(direction)

        reward = self.grid[self.agent.x][self.agent.y].reward() - GridWorld.PENALTY

        self.total_reward += reward

        return reward

    @staticmethod
    def generate(entities, width, height):
        grid = [[None for _ in range(height)] for _ in range(width)]

        count = 0

        for key, value in entities.iteritems():
            if key is not Empty:
                count += value

        assert count < width * height

        entities[Empty] = width * height - count

        indices = [(x, y) for x in range(width) for y in range(height)]
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

    def __init__(self, grid_world, x, y):
        self.grid_world = grid_world
        self.x = x
        self.y = y

    def id(self):
        return self.__class__.ID

    def color(self):
        return self.__class__.COLOR


class Agent(Entity):
    COLOR = '#B873FF'

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


class Goal(Block):
    REWARD = 1
    TERMINAL = True
    COLOR = '#49732C'


class Coin(Block):
    REWARD = 0.1
    COLOR = '#FFD225'

    def __init__(self, grid, x, y):
        super(Coin, self).__init__(grid, x, y)

        self.active = True

    def id(self):
        return self.__class__.ID if self.active else Empty.ID

    def color(self):
        return self.__class__.COLOR if self.active else Empty.COLOR

    def reward(self):
        return self.__class__.REWARD if self.active else Empty.REWARD

    def interact(self, agent):
        super(Coin, self).interact(agent)

        self.active = False


class Water(Block):
    REWARD = -0.1
    COLOR = '#0A73FF'


class Fire(Block):
    REWARD = -1
    TERMINAL = True
    COLOR = '#B20702'


class Wall(Block):
    COLOR = '#422C25'

    def interact(self, agent):
        pass


class Portal(Block):
    COLOR = '#481B5E'

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

    def __init__(self, grid, x, y):
        super(Switch, self).__init__(grid, x, y)

        self.active = True

    def id(self):
        return self.__class__.ID if self.active else Empty.ID

    def color(self):
        return self.__class__.COLOR if self.active else Empty.COLOR

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

    def __init__(self, grid, x, y):
        super(Door, self).__init__(grid, x, y)

        self.open = False

    def id(self):
        return self.__class__.ID if not self.open else Empty.ID

    def color(self):
        return self.__class__.COLOR if not self.open else Empty.COLOR

    def interact(self, agent):
        if self.open:
            agent.x = self.x
            agent.y = self.y


class Display:
    def __init__(self, width=16, height=16, field_size=16):
        pygame.init()

        self.width = width
        self.height = height
        self.field_size = field_size
        self.screen = pygame.display.set_mode((self.width * field_size, self.height * field_size))

    def xy2rect(self, x, y):
        return pygame.Rect(x * self.field_size + 1, y * self.field_size + 1, self.field_size - 2, self.field_size - 2)

    def draw(self, grid_world):
        self.screen.fill((255, 255, 255))

        for x in range(self.width):
            for y in range(self.height):
                pygame.draw.rect(self.screen, pygame.Color(grid_world.color(x, y)), self.xy2rect(x, y))

        pygame.display.update()


if __name__ == '__main__':
    gw = GridWorld(entities={Goal: 1, Wall: 15, Switch: 1, Door: 3, Portal: 3, Fire: 2, Water: 10, Coin: 10})
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
