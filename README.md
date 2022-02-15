# Introduction

This is the application environment for DHB experiments. 
To build all experiments you require Simexpal to be 
installed.
For some of the defined experiments you will need some local 
graph files to exist. Please add them to the `/instances`
folder (or create symlinks).

# Requirements

Building the `dhb_exp` target you require C++14, CMake and OpenMP.

# Setup

## Simex

```
$ pip3 install simexpal
```

# Simex

Copy the `launchers.yml` file to ~/.simexpal (and create folder if it does not exist yet).

## Build

`$ simex develop`
