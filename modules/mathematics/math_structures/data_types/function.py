import numpy as np

class Function(object):

    def __init__(self,function_type):
        self.domain = None
        self.range = None
        self.function = None
        self.function_type = function_type
        self.order = None

    def initialize(self):

        if self.function_type == 'polynomial':
            return None
        return

    
