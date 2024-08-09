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

import math

class RenderGrid():    
  def __init__(self, image_height, image_width, tile_width, tile_height):
    self.image_height = image_height
    self.image_width = image_width
    self.tile_height = tile_height
    self.tile_width = tile_width
    self.grid_height = math.ceil(image_height / tile_height)
    self.grid_width = math.ceil(image_width  / tile_width)
