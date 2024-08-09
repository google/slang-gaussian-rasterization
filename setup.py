# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import setup, find_packages

setup(
    name='slang_gaussian_rasterization',
    version='0.1.0',    
    description='A Slang.D implementation of the CUDA acclerated rasterizer that is described in the \"3D Gaussian Splatting for Real-Time Rendering of Radiance Fields\", Kerbl and Kopanas 2023',
    author='George Kopanas',
    author_email='gkopanas@google.com',
    license='Apache 2.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'slang_gaussian_rasterization': ['slang_gaussian_rasterization/internal/slang/alpha_blend_sai.slang']
    },
    install_requires=['slangtorch',
                      'torch']
)

