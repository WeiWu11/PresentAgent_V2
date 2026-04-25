# Topic 20: What is Flow Matching? - by Alyona Vert.

Today, we’re exploring Flow Matching (FM), a concept that might sound complex but is more approachable than it seems. If it feels overwhelming at first, don’t worry – by the end of this episode, you’ll have a clear understanding of its key ideas and practical applications.

Why is Flow Matching worth discussing now? It’s gaining attention for its role in top generative models like Flux (text-to-image), F5-TTS and E2-TTS (text-to-speech), and Meta’s MovieGen (text-to-video). These models consistently achieve state-of-the-art results, and some experts argue that FM might even surpass diffusion models. But why is that the case?

FM enhances Continuous Normalizing Flows (CNFs), a framework for generating realistic samples of complex data – whether images, audio, or text – starting from structured noise. While powerful, CNFs face challenges such as long training times and intricate techniques for speeding up sampling. Flow Matching tackles these issues by optimizing the path from noise to structured outputs, streamlining CNFs and reducing the inefficiencies caused by differential equation computations. Put simply, FM focuses on learning how to match flows of probability distributions over time.

Still sounds tricky? Let’s break it all down, examine the details, and provide real-world examples of its implementation so you can see its potential in action. Let’s get started!
