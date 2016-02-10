from enum import Enum


Direction = Enum('Direction', 'none left right up down')


class GridWorld:
    def __init__(self, grid=None, width=10, height=10, entities=None, display=False):
        pass

    def state(self):
        pass


class CounterMetaClass(type):
    counter = 0

    def __new__(mcs, name, bases, attributes):
        attributes['id'] = CounterMetaClass.counter
        CounterMetaClass.counter += 1

        return type.__new__(mcs, name, bases, attributes)


class Entity:
    __metaclass__ = CounterMetaClass

    def __init__(self, grid, x, y):
        self.grid = grid
        self.x = x
        self.y = y

    def id(self):
        return self.id


class Agent(Entity):
    def move(self, direction):
        pass


class Block(Entity):
    reward = 0
    terminal = False
    color = '#000000'

    def __init__(self, grid, x, y):
        super(Block, self).__init__(grid, x, y)

        self.reward = self.__class__.reward
        self.terminal = self.__class__.terminal
        self.color = self.__class__.color

    def interact(self, block):
        pass


class Goal(Block):
    pass


class Water(Block):
    pass


class Wall(Block):
    pass


class Portal(Block):
    pass


class Switch(Block):
    pass


class Door(Block):
    pass

