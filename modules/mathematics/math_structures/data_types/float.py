import numpy as np

class Float(object):

    def __init__(self,float_or_int_val,float_type=np.float64):
        self.float64 = False
        self.float32 = False
        self.float16 = False
        self.type = None
        self.val = None
        
        if float_type != None:
            self.type = float_type
            if type(float_type(1)) == type(np.float64(1)):
                self.val = float_type(float_or_int_val)
                self.float64 = True
            elif type(float_type(1)) == type(np.float32(1)):
                self.val = float_type(float_or_int_val)
                self.float32 = True
            else:
                self.val = float_type(float_or_int_val)
                self.float16 = True
        else:
            self.val = Float(float_or_int_val.val)
        
    def __str__(self):
        return str(self.val)

    def __repr__(self):
        return str(self.val)

    def __eq__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return self.val == other
        return self.val == other.val and type(self.type(1)) == type(other.type(1))

    def __ne__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return self.val != other
        return self.val != other.val or type(self.type(1)) != type(other.type(1))

    def __lt__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return self.val < other
        return self.val < other.val

    def __gt__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return self.val > other
        return self.val > other.val

    def __ge__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return self.val <= other
        return self.val <= other.val

    def __le__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return self.val >= other
        return self.val >= other.val
    
    def __add__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return Float(self.val + other)
        return Float(self.val + other.val)

    def __sub__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return (self.val - other)
        return Float(self.val - other.val)

    def __mul__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return Float(self.val * other)
        return Float(self.val * other.val)
    
    def __truediv__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return Float(self.val / other)
        return Float(self.val / other.val)

    def __radd__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return Float(self.val + other)
        return Float(self.val + other.val)

    def __rsub__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return Float(self.val - other)
        return Float(self.val - other.val)

    def __rmul__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return Float(self.val * other)
        return Float(self.val * other.val)
    
    def __rtruediv__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return Float(other / self.val)
        return Float(other.val / self.val)
