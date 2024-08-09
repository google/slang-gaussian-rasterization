# Slang.D Gaussian Splatting Rasterizer

This repository contains a Slang.D implementation of the CUDA accelerated rasterizer that is described in the [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). We provide only the rasterizer and API bindings to the most popular implementations of the complete 3D Gaussian Splatting implementation: 1) the [original inria code-base](https://github.com/graphdeco-inria/gaussian-splatting) and 2) [gsplat from nerf-studio](https://github.com/nerfstudio-project/gsplat). 

*But why Slang.D?*

Slang.D is a *unified platform for real-time, inverse and differentiable rendering*. Slang serves as an open-source languge that allows developers to maintain a single code-base for differentiable high-perfomance rendering code-bases which can compile down to redenring code for different platforms like D3D, Vulkan, OpenGL, OptiX, CUDA etc.

This allows for a **single code-base** of the rendering code than runs both under the differentiable framework of your choice (i.e Pytorch or Jax) and the actual renderer that can be released for production (i.e OpenGL). This makes significant strides toward maintainability and reduces the likelihood of errors.

On top of that and potentially the most imporant for *research* is the ability of Slang.D to **differentiate automatically** through complicated rendering kernerls that include *arbitrary control flow, user-defined types, dynamic dispatch, generics, and global memory accesses.*



## Installing for Users [Not ready yet!]

If you only care about using this library and you do not want to modify it's contents the most convenient way is to simply pip install it:

```bash
pip install slang-gaussian-rasterization
```

## Installing for Developers

If you want to modify the content of this library and you want to experiment with the renderer you need to install the pip package in development mode.

```bash
git clone [placeholder]
cd ./slang-gaussian-rasterization
pip install -e .
```

Now while you can use this pip package as any other python package, any change you do either on the python side or on the Slang.D side will automatically reflect in your installed package without the need to pip install every time you make a change.

## Using it with popular 3DGS optimization libraries

To use the renderer during training we need to plug this in either the original inria implementation or nerf-studio:

**Original Inria Implementiation:** 

To run the original inria training code with the slang back-end we need to patch [train.py](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/train.py). The following commands asssume that your 3dgs-inria code base is at [path-to-3dgs-inria]

```bash
cd [path-to-3dgs-inria]
wget [place-holder-for-path-to-patch]
git am 3dgs.patch
```

Now you can run the train.py scripts as described in the inria 3DGS repo and the renderer you will be using is from the slang-gaussian-rasterization package.

**gsplat Implementiation:** 
[TODO]

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.

## Disclaimer

This project is not an official Google project. It is not supported by
Google and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.
