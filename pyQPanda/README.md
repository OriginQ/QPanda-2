# How to use

1. (Python==3.6) there is no need to update .pyd library. You can use it directly

2. (Python!=3.6) First compile the QPanda project, and replace pyqpanda/pyQPanda.pyd with the release library.
    - Before compiling the QPanda project, the python path must be configured first.
    - Add the $(PythonPath)/include to your includepath, and $(PythonPath)/libs to your library path
    - Compile and obtain .pyd file

3. The Config.xml and MetadataConfig.xml must be placed right at the root directory of your python script

# Brief

pyqpanda : QPanda Basic API
pyqpanda.utils : Extended QPanda API
pyqpanda.Algorithm : pyqpanda algorithm pack
pyqpanda.Algorithm.demo: Some algorithm demonstration
pyqpanda.Algorithm.test: Test for pyqpanda.Algorithm
pyqpanda.Algorithm.fragments: Some algorithm fragments
pyqpanda.Hamiltonian : pyqpanda Hamiltonian utilities
