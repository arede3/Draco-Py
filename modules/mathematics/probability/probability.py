class Probability(object):

    def __init__(self, probability_function, *args, **kwargs):

        self.probability_function = probability_function
        self.args = args
        self.kwargs = kwargs
        self.ans = None

        if probability_function == None or args == None or kwargs == None:
            self.probability_function = None
            self.args = None
            self.kwargs = None

        if args == ():
            self.probability_function = probability_function
            self.args = args
            self.kwargs = kwargs
            self.ans = self.probability_function(**kwargs)

        if kwargs == {}:
            self.probability_function = probability_function
            self.args = args
            self.kwargs = kwargs
            self.ans = self.probability_function(*args)

        raise Exception('Please provide proper inputs!')

    def
