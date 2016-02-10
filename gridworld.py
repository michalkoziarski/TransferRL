from enum import Enum


Direction = Enum('Direction', 'none left right up down')


class GridWorld:
    penalty = 0.01

    def __init__(self, grid=None, width=10, height=10, entities=None, display=False):
        if grid:
            self.grid = grid
        else:
            self.grid = GridWorld.generate(entities, width, height)

        for x in range(len(self.grid)):
            for y in range(len(self.grid[0])):
                self.grid[x][y] = self.grid[x][y](self.grid, x, y)

        self.penalty = GridWorld.penalty
        self.display = display

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

    def __init__(self, grid, x, y):
        self.grid = grid
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

