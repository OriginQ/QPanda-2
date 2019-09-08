g++的编译命令为

````
g++ -std=c++14  test.cpp  -I{安装目录}/include -I{安装目录}/include/ThirdParty/ -L{安装目录}/lib/  -lQPanda2 -lTinyXML -fopenmp -o test
````
