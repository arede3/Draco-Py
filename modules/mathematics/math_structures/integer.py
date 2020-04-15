import numpy as np

class Integer(object):

    def __init__(self,float_or_int_val,int_type=np.int64):
        self.int64 = False
        self.int32 = False
        self.int16 = False
        self.int8 = False
        self.int0 = False
        self.type = int_type
        self.val = None
        
        if type(int_type(1)) == type(np.int64(1)):
            self.val = int_type(float_or_int_val)
            self.int64 = True
        elif type(int_type(1)) == type(np.int32(1)):
            self.val = int_type(float_or_int_val)
            self.int32 = True
        elif type(int_type(1)) == type(np.int16(1)):
            self.val = int_type(float_or_int_val)
            self.int16 = True
        elif type(int_type(1)) == type(np.int8(1)):
            self.val = int_type(float_or_int_val)
            self.int8 = True
        else:
            self.val = int_type(float_or_int_val)
            self.int0 = True

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
            return Integer(self.val + other)
        return Integer(self.val + other.val)

    def __sub__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return Integer(self.val - other)
        return Integer(self.val - other.val)

    def __mul__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return Integer(self.val * other)
        return Integer(self.val * other.val)
    
    def __truediv__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return Integer(self.val / other)
        return Integer(self.val / other.val)

    def __radd__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return Integer(self.val + other)
        return Integer(self.val + other.val)

    def __rsub__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return Integer(self.val - other)
        return Integer(self.val - other.val)

    def __rmul__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return Integer(self.val * other)
        return Integer(self.val * other.val)
    
    def __rtruediv__(self,other):
        if type(other) != type(self) and (type(other) == type(float(1)) or type(other) == type(int(1)) or type(other) == type(self.type(1))):
            return Integer(other / self.val)
        return Integer(other.val / self.val)
