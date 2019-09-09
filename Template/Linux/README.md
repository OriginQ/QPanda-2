g++的编译命令为
````

宿主机有libcurl时
g++ test.cpp -std=c++14 -fopenmp -I{QPanda安装路径}/include/qpanda2/ -I{QPanda安装路径}/include/qpanda2/ThirdParty/ -L{QPanda安装路径}/lib/ -lQPanda2 -lTinyXML -lcurl -o test

宿主机没有libcurl时
g++ test.cpp -std=c++14 -fopenmp -I{QPanda安装路径}/include/qpanda2/ -I{QPanda安装路径}/include/qpanda2/ThirdParty/ -L{QPanda安装路径}/lib/ -lQPanda2 -lTinyXML -o test
````

使用MPI并行计算时的编译命令
````

宿主机有libcurl时
mpic++ test.cpp -std=c++14 -fopenmp -I{QPanda安装路径}/include/qpanda2/ -I{QPanda安装路径}/include/qpanda2/ThirdParty/ -L{QPanda安装路径}/lib/ -lQPanda2 -lTinyXML -lcurl -o test

宿主机没有libcurl时
mpic++ test.cpp -std=c++14 -fopenmp -I{QPanda安装路径}/include/qpanda2/ -I{QPanda安装路径}/include/qpanda2/ThirdParty/ -L{QPanda安装路径}/lib/ -lQPanda2 -lTinyXML -o test
````
