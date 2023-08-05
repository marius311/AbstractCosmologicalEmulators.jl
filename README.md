# AbstractCosmologicalEmulators.jl

[![Build status (Github Actions)](https://github.com/CosmologicalEmulators/AbstractCosmologicalEmulators.jl/workflows/CI/badge.svg)](https://github.com/CosmologicalEmulators/AbstractCosmologicalEmulators.jl/actions)
[![codecov](https://codecov.io/gh/CosmologicalEmulators/AbstractCosmologicalEmulators.jl/branch/main/graph/badge.svg?token=0PYHCWVL67)](https://codecov.io/gh/CosmologicalEmulators/AbstractCosmologicalEmulators.jl)
![size](https://img.shields.io/github/repo-size/CosmologicalEmulators/AbstractCosmologicalEmulators.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)

`AbstractCosmologicalEmulators.jl` is the central `Julia` package within the the [CosmologicalEmulators](https://github.com/CosmologicalEmulators) Github organization, which defines methods and structs used by the other packages hosted by the organization.

In this moment the emulators here used are based only on the [`SimpleChains.jl`](https://github.com/PumasAI/SimpleChains.jl) library, whose performance is excellent on the CPU for the kind of small neural networks (NN) that we employ. We plan to include other frameworks, such as [`Lux.jl`](https://github.com/LuxDL/Lux.jl), in order to support models running on the GPU. If you want include a new NN/GP framework, feel free to open a PR or write a message.

## Roadmap to v1.0.0

Step | Status| Comment
:------------ | :-------------| :-------------
Interface with `SimpleChains.jl` | :heavy_check_mark: | Implemented
Support for vectorization | :white_check_mark: | Implemented, needs some polishing
Interface with `Lux.jl` | :construction: | WIP
GPU support | :construction: | WIP
Stable API | :construction: | WIP

## Authors

- [Marco Bonici](https://www.marcobonici.com), PostoDoctoral Researcher at INAF-IASF Milano
