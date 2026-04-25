# What is Flow Matching? Generative AI Guide | Ultralytics

At its heart, flow matching simplifies the generation process by focusing on the velocity of data transformation rather than just the marginal probabilities. This method draws inspiration from continuous normalizing flows but avoids the high computational cost of calculating exact likelihoods.

Vector Fields : The central component of flow matching is a neural network that predicts a velocity vector for any given point in space and time. This vector tells the data point which direction to move to become a realistic sample.

Optimal Transport: Flow matching often aims to find the most efficient path to transport mass from one distribution to another. By minimizing the distance traveled, models can achieve faster inference times. Techniques like optimal transport help define these straight paths, ensuring that noise maps to data in a geometrically consistent way.

Conditional Generation: Similar to how Ultralytics YOLO26 conditions detections on input images, flow matching can condition generation on class labels or text prompts. This allows for precise control over the generated content, a key feature in modern text-to-image and text-to-video pipelines.
