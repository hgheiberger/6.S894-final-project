# Project Name: 
Comparing GPU-Accelerated Ray-Tracing Renderers

# Team Members: 
Harry Heiberger, Henry Heiberger

# Description: 
For our project, we plan to explore ray-tracing rendering by implementing and comparing a series of GPU-accelerated ray-tracing pipelines to render a simple scene containing various reflective objects.  Starting with a straightforward CPU implementation as a baseline, we plan to implement both a pure CUDA ray-tracing renderer and one that takes advantage of the CUDA OptiX API.  If time permits, we may also attempt to implement a ray-tracing renderer using Vulkan but we’re keeping this as a stretch goal due to the added complexity of switching to GLSL versus CUDA C++.  Following implementation, we plan to compare our ray-tracing renderers both qualitatively through metrics such as ease of implementation and code clarity and quantitatively through performance benchmarks.  We hope to leave with a strong understanding of the ray tracing problem and the advantages and disadvantages of some of the various tools available to pursue it.

# Resources: 
Both Henry and I do not have access to any additional GPU resources so we will develop and benchmark our kernels using the classes provided RTX A4000 GPU.  We expect this resource to work fine for our graphics project. 
