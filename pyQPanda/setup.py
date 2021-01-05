from setuptools import setup, find_packages  
import os
  
# Read in requirements.txt
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

README_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                           'README.md')
with open(README_PATH) as readme_file:
    README = readme_file.read()

setup(  
    name = "pyqpanda",  
    version = "3.7.3",  
    license = "Apache Licence",  
    author = "OriginQ",
    install_requires=requirements,
    description="pyQPanda is Python wrapper of QPanda.",    
    packages = find_packages(),  
    py_modules = ['psi4_wrapper'],
    #data_files=[(['psi4_wrapper.py'])],
    include_package_data = True,  
    classifiers=[
	"Development Status :: 4 - Beta",
	"Operating System :: MacOS :: MacOS X",
	"Operating System :: Microsoft :: Windows :: Windows 10",
	"Operating System :: POSIX :: Linux",
	"Topic :: Software Development :: Libraries :: Python Modules",
	"Programming Language :: Python",
	"Programming Language :: Python :: 3.5",
	"Programming Language :: Python :: 3.6",
	"Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
	],
)