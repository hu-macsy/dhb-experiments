# Introduction

This is the application environment for DHB experiments. 
To build all experiments you must install
[Simexpal](https://github.com/hu-macsy/simexpal).

For some of the defined experiments you will need some local 
graph files to exist. Please add them to the `/instances`
folder (or create symlinks).

# Requirements

## C++

Building the `dhb_exp` target you require C++14, CMake and OpenMP.

## Python

Install all requirements present in `requirements.txt`. To run the evaluation
script.

```
$ pip3 install -r requirements.txt
```
