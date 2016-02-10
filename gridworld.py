import pygame

from enum import Enum


Direction = Enum('Direction', 'none left right up down')


class GridWorld:
    penalty = 0.01

    def __init__(self, grid=None, width=10, height=10, entities=None):
        if grid:
            self.grid = grid
        else:
            self.grid = GridWorld.generate(entities, width, height)

        self.width = len(self.grid)
        self.height = len(self.grid[0])

        for x in range(self.width):
            for y in range(self.height):
                self.grid[x][y] = self.grid[x][y](self, x, y)

        self.penalty = GridWorld.penalty

    def state(self):
        pass

    @staticmethod
    def generate(entities, width, height):
        pass


class CounterMetaClass(type):
    counter = 0

    def __new__(mcs, name, bases, attributes):
        attributes['id'] = CounterMetaClass.counter
        CounterMetaClass.counter += 1

        return type.__new__(mcs, name, bases, attributes)


class Entity:
    __metaclass__ = CounterMetaClass

    color = '#000000'

    def __init__(self, grid_world, x, y):
        self.grid_world = grid_world
        self.x = x
        self.y = y
        self.color = self.__class__.color

    def id(self):
        return self.id


class Agent(Entity):
    color = '#B873FF'

    def move(self, direction):
        pass


class Block(Entity):
    reward = 0
    terminal = False

    def __init__(self, grid, x, y):
        super(Block, self).__init__(grid, x, y)

        self.reward = self.__class__.reward
        self.terminal = self.__class__.terminal

    def interact(self, block):
        pass


class Empty(Block):
    color = '#D4D1CA'


class Goal(Block):
    reward = 1
    terminal = True
    color = '#FFD225'


class Water(Block):
    reward = -0.1
    color = '#0A73FF'


class Fire(Block):
    reward = -1
    terminal = True
    color = '#B20702'


class Wall(Block):
    color = '#422C25'


class Portal(Block):
    color = '#481B5E'


class Switch(Block):
    color = '#218B87'


class Door(Block):
    color = '#B63F00'


class Display:
    def __init__(self, width=10, height=10, field_size=30):
        pygame.init()

        self.width = width
        self.height = height
        self.field_size = field_size
        self.screen = pygame.display.set_mode((self.width * field_size, self.height * field_size))

    def xy2rect(self, x, y):
        return pygame.Rect(x * self.field_size + 1, y * self.field_size + 1, self.field_size - 2, self.field_size - 2)

    def draw(self, grid):
        self.screen.fill((255, 255, 255))

        for x in range(self.width):
            for y in range(self.height):
                pygame.draw.rect(self.screen, grid[x][y].color, self.xy2rect(x, y))

        pygame.display.update()
