import os
from setuptools import setup

curdir = os.getcwd()
sdir = os.path.dirname(os.path.abspath(__file__))
codedir = os.sep.join([sdir, 'spectool'])
os.chdir(codedir)
os.system('make')
os.chdir(curdir)

setup()
