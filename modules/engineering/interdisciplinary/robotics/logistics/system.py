import sys
import os

def mkdir(path):
    directory = os.mkdir(path)
    sys.path.append(directory)
    return 'Created to %s.' % (str(directory))

def cd(path):
    os.chdir(path)
    return 'Changed to %s.' % (str(path))

def cwd():
    directory = os.getcwd()
    sys.path.append(directory)
    return 'Currently in %s.' % (str(directory))
    
