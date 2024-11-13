from setuptools import setup, find_packages
import platform  
import os
  
# Read in requirements.txt
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'README.md')
with open(README_PATH) as readme_file:
    README = readme_file.read()

is_win = (platform.system() == 'Windows')
if is_win:
    pd_files = ['*.pyd', '*.dll', '*.pyi']
else :
    pd_files = ['*.so', '*.pyi']


setup(  
    name = "pyqpanda",  
    version = "3.8.4",  
    license = "Apache Licence",  
    author = "OriginQ",
    install_requires=requirements,
    description="pyQPanda is Python wrapper of QPanda.",    
    packages = find_packages(),
    
    py_modules = ['psi4_wrapper'],
    package_data={
        '':pd_files
    },
    include_package_data = True,  
    classifiers=[
	"Development Status :: 4 - Beta",
	"Operating System :: MacOS :: MacOS X",
	"Operating System :: Microsoft :: Windows :: Windows 10",
	"Operating System :: POSIX :: Linux",
	"Topic :: Software Development :: Libraries :: Python Modules",
	"Programming Language :: Python",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
	],
)