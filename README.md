# Slang.D Gaussian Splatting Rasterizer

This repository contains a Slang.D implementation of the CUDA acclerated rasterizer that is described in the [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). We provide only the rasterizer and API bindings to the most popular implementations of the complete 3D Gaussian Splatting implementation, the original inria code-base and [gsplat from nerf-studio](https://github.com/nerfstudio-project/gsplat). We open source this extra library because the Slang.D framework allows for mostly automatic differentiation of the renderer something we consider a valuable tool for accelerating research.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.

## Disclaimer

This project is not an official Google project. It is not supported by
Google and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.
