
import os
from datetime import datetime


class MyConfig:
    def __init__(self):
        self.data = {}
    
    def __str__(self):
        return str(self.data)
    
    def __repr__(self):
        return str(self.data)

GLOBAL_CONFIG = MyConfig()


def get_config(key, default=None):
    return GLOBAL_CONFIG.data.get(key, default)


def set_config(key, value):
    GLOBAL_CONFIG.data[key] = value


def set_from_dict(config_dict):
    GLOBAL_CONFIG.data = config_dict


def reset_config():
    GLOBAL_CONFIG.data = {}


def get_config_dict():
    return GLOBAL_CONFIG.data

def print_config():
    print(GLOBAL_CONFIG.data)
