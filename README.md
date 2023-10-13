# HaLRTC-in-Julia-1.x
HaLRTC is an tensor completion proposed in 2013 which is commonly used as a baseline for more advanced tensor completion techniques. The original paper can be found [here](doi.org/10.1109/TPAMI.2012.39).

Previously, [xinychen's codebase](https://github.com/xinychen/tensor_completion) has implemented HaLRTC in Julia 0.x, but the code has numerous compatibility issues with the newest Julia, making it very hard to migrate. Hence, as a part of a course project, I wrote this simple demo of HaLRTC in Julia 1.x. The code tests HaLRTC on 12 standard testing images, and the included function can be easily extracted for use in your own Julia program.

Dependencies:
+ Plots
+ Images
+ FileIO
+ LinearAlgebra
+ SchattenNorms
+ Statistics
+ DelimitedFiles
