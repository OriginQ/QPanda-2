from setuptools import setup, find_packages  
  
# Read in requirements.txt
requirements = open('requirements.txt').readlines()
requirements = [r.strip() for r in requirements]

setup(  
    name = "pyqpanda",  
    version = "3.2.7",  
    license = "Apache Licence",  
    author = "OriginQ",
    install_requires=requirements,
    description="pyQPanda is Python wrapper of QPanda.",
    long_description=open("README.rst",encoding='utf-8').read(),    
    packages = find_packages(),  
    include_package_data = True,  
    classifiers=[
	"Development Status :: 4 - Beta",
	"Operating System :: MacOS :: MacOS X",
	"Operating System :: Microsoft :: Windows :: Windows 10",
	"Operating System :: POSIX :: Linux",
	"Topic :: Software Development :: Libraries :: Python Modules",
	"Programming Language :: Python",
	"Programming Language :: Python :: 3.6",
	],
)
