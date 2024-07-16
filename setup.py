import os
import glob
from setuptools import setup

curdir = os.getcwd()
sdir = os.path.dirname(os.path.abspath(__file__))
codedir = os.sep.join([sdir, 'spectool'])
os.chdir(codedir)
# sharefiles = glob.glob('*.so')
# for f in sharefiles:
#     # remove old .so files before building, avoid to use the old .so files
#     os.remove(f)

python_version = 'python' + str(os.sys.version_info[0]) + '.' + str(os.sys.version_info[1])
os.environ['PYTHON_VERSION'] = python_version
print(f"python version is set to: {os.environ['PYTHON_VERSION']}")
os.system('make clean')
os.system('make')
os.chdir(curdir)

setup()
