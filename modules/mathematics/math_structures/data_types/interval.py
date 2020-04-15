import numpy as np

class Interval(object):

    def __init__(self,min_val=-np.inf,max_val=np.inf):
        self.interval = None
        self.left = min_val
        self.right = max_val
        self.left_inclusive = None
        self.right_inclusive = None

    def __str__(self):
        
        if self.left_inclusive == True and self.right_inclusive == True:
            self.interval = '[%s,%s]' % (self.left, self.right)
        elif self.left_inclusive == True and self.right_inclusive == False:
            self.interval = '[%s,%s)' % (self.left, self.right)
        elif self.left_inclusive == False and self.right_inclusive == True:        
            self.interval = '(%s,%s]' % (self.left, self.right)
        else:
            self.interval = '(%s,%s)' % (self.left, self.right)

        return self.interval

    def __repr__(self):
        return str(self)

    def set_inclusiveness(self,left_inclusive=False,right_inclusive=False):
        self.left_inclusive = left_inclusive
        self.right_inclusive = right_inclusive
    
    

    
