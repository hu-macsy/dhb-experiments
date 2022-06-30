# Introduction

This is the application environment for DHB experiments. 

We define all our experiments using
[Simexpal](https://github.com/hu-macsy/simexpal) (see `experiments.yml`), thus
you must install Simexpal to build and execute all experiments.

# Requirements

## Instances

Most instances can be installed using:

```shell
$ simex instances install
```

Please run the `cleanup.sh` script after the Simexpal instance installation step 
to execute necessary graph file conversion steps.  

Some of the defined experiments require graph files which are not downloaded
using the instances specification of Simexpal. Please copy all as repository
`local` specified graph files (except the set `generated`) to the `/instances`
folder manually. 

## C++

Building the `dhb_exp` target you require C++14, CMake and OpenMP.

## Python

To run the evaluation script install all requirements present in
`requirements.txt`. 

```shell
$ pip3 install -r requirements.txt
```
