# Flow Matching Explained: From Noise to Robot Actions | Federico Sarrocco

Flow Matching is grounded in the concept of learning a velocity field (also known as a vector field). This velocity field defines a flow ψ t \psi_t ψ t ​ by solving an ordinary differential equation (ODE) through simulation. Essentially, a flow is a deterministic, time-continuous, bijective transformation of the d-dimensional Euclidean space R d \mathbb{R}^d R d .

The primary objective is to construct a flow that transforms a sample X 0 ∼ p \mathbf{X_0} \sim \mathbf{p} X 0 ​ ∼ p from a source distribution p \mathbf{p} p into a target sample X 1 = ψ 1 ( X 0 ) \mathbf{X_1} = \psi_1(\mathbf{X_0}) X 1 ​ = ψ 1 ​ ( X 0 ​ ) such that X 1 ∼ q \mathbf{X_1} \sim \mathbf{q} X 1 ​ ∼ q , where q \mathbf{q} q is the desired target distribution.

More specifically, the goal is to find the parameters of the flow defined as a learnable velocity u t θ \mathbf{u}_t^\theta u t θ ​ that generates intermediate distributions p t \mathbf{p}_t p t ​ with p 0 = p \mathbf{p}_0 = \mathbf{p} p 0 ​ = p and p 1 = q \mathbf{p}_1 = \mathbf{q} p 1 ​ = q for each t t t in ( 0 , 1 ) (0,1) ( 0 , 1 ) . This ensures a smooth transition from the source to target distribution.

The Flow Matching Loss : The core of FM is the Flow Matching loss. It measures the difference between the learned velocity field and the ideal velocity field that would perfectly generate the probability path. By minimizing this loss, the model learns to guide the source distribution along the desired path towards the target data distribution.

Flow models, introduced to the machine learning community as Continuous Normalizing Flows (CNFs), initially involved maximizing the likelihood of training examples by simulating and differentiating the flow during training. However, due to computational challenges, subsequent research focused on learning these flows without simulation, leading to modern Flow Matching algorithms.
