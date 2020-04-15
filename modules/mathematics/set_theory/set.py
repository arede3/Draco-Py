import random as r

class Set(object):

    def __init__(self, iterable):
        self.S = list(iterable)
        self.P_S = list()

    def factorial(self,n):

        if n <= 0:
            return 1

        return n*self.factorial(n-1)

    def nCr(self,n,r):
        return int(self.factorial(n)/(self.factorial(r)*self.factorial(n-r)))

    def nPr(self,n,r):
        return int(self.factorial(n)/self.factorial(n-r))

    def find_subsets(self):
        subsets = {}

        l = len(self.S)

        for i in range(0,l+1):
            p = i
            if i == 0:
                subsets[i] = [Set([])]
                continue
            elif i == l:
                subsets[i] = [Set(self.S)]
                break
            else:
                num_of_choices = self.nCr(l,i)
                print(num_of_choices)
                i_element_subsets = []
                for j in range(num_of_choices):
                    if p < l: 
                        i_element_set = list()   
                        for k in range(p):
                            if self.S[k] not in i_element_set:
                                i_element_set.append(self.S[k])
                        p += 1
                    else:
                        break
                    if i_element_set not in i_element_subsets:
                        i_element_subsets.append(i_element_set)
                subsets[i] = i_element_subsets
                

        return subsets

    def cardinality(self):
        return len(self.S)
    
    def power_set(self):
        self.P_S = self.find_subsets()
        return list(self.P_S)

    

    def __str__(self):
        s = '{'
        l = len(self.S)
        for val in self.S:
            if val != self.S[l-1]:
                s += (str(val)+', ')
            else:
                s += str(val)
        s += '}'

        return s
