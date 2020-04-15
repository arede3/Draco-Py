import numpy as np

class Function(object):

    def __init__(self,function,domain):
        self.domain = domain
        self.range = function(domain)

    
