import os
import glob
from setuptools import setup
import subprocess

curdir = os.getcwd()
sdir = os.path.dirname(os.path.abspath(__file__))
codedir = os.sep.join([sdir, 'spectool'])
os.chdir(codedir)
# sharefiles = glob.glob('*.so')
# for f in sharefiles:
#     # remove old .so files before building, avoid to use the old .so files
#     os.remove(f)


def check_gsl():
    try:
        output = subprocess.check_output(['pkg-config', '--modversion', 'gsl'], text=True)
        installed_version = output.strip()
        required_version = '2.7'
        if installed_version < required_version:
            raise Exception(f'gsl version < {required_version}')
    except subprocess.CalledProcessError:
        raise ValueError('gsl is not installed')
    print('gsl version =', installed_version)


def check_fftw3():
    try:
        output = subprocess.check_output(['pkg-config', '--modversion', 'fftw3'], text=True)
        installed_version = output.strip()
        # required_version = '3.3.10'
        # if installed_version < required_version:
        #     raise Exception(f'fftw3 version < {required_version}')
    except subprocess.CalledProcessError:
        raise ValueError('fftw3 is not installed')
    print('fftw3 version =', installed_version)


def check_pybind11():
    try:
        import pybind11
    except ImportError:
        raise ValueError('pybind11 is not installed')
    print('pybind11 is installed')


check_gsl()
check_fftw3()
check_pybind11()
python_version = 'python' + str(os.sys.version_info[0]) + '.' + str(os.sys.version_info[1])
os.environ['PYTHON_VERSION'] = python_version
print(f"python version is set to: {os.environ['PYTHON_VERSION']}")
os.system('make clean')
os.system('make')
os.chdir(curdir)

setup()
