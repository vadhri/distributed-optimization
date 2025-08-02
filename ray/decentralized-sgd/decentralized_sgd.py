import ray
import torch
import matplotlib.pyplot as plt

ray.shutdown()
ray.init(include_dashboard=True, num_cpus=10)

@ray.remote
class Worker:
    def __init__(self, id, X, y):
        self.id = id
        self.X = X
        self.y = y
        self.neighbors = []
        self.w = torch.randn(2, requires_grad=False)
        self.lr = 0.1
        self.received_weights = []

    def set_neighbors(self, neighbors):
        self.neighbors = neighbors
        self.received_weights = [self.w for _ in neighbors]  # Initialize with own weights

    def sgd_step(self):
        y_hat = self.X @ self.w
        grad = 2 * (y_hat - self.y) @ self.X / len(self.y)
        self.w = self.w - self.lr * grad

    def send_weights(self):
        futures = []
        for i, neighbor in enumerate(self.neighbors):
            futures.append(neighbor.receive_weights.remote(self.w, self.id))
        return futures

    def receive_weights(self, w, sender_id):
        idx = sender_id % len(self.neighbors)
        self.received_weights[idx] = w
        return True

    def update_weights(self):
        all_ws = [self.w] + self.received_weights
        self.w = sum(all_ws) / len(all_ws)

    def get_param(self):
        return self.w

    def compute_loss(self):
        y_hat = self.X @ self.w
        return torch.mean((y_hat - self.y) ** 2).item()

def main():
    N = 100
    x = torch.linspace(-1, 1, N).unsqueeze(1)
    true_w = torch.tensor([2.0, -3.0])
    X = torch.cat([x, torch.ones_like(x)], dim=1)
    y = X @ true_w + 0.1 * torch.randn(N)

    cpus = int(ray.available_resources().get('CPU', 1))
    print(f"Using {cpus} workers (CPUs available: {ray.available_resources().get('CPU', 1)})")

    parts = torch.chunk(X, cpus)
    labels = torch.chunk(y, cpus)

    workers = [Worker.remote(i, parts[i], labels[i]) for i in range(cpus)]

    for i, worker in enumerate(workers):
        neighbors = [workers[(i + 1) % len(workers)]]  # simple ring topology
        ray.get(worker.set_neighbors.remote(neighbors))

    # Store history for plotting: weights and losses per worker per step
    weights_history = [[] for _ in range(cpus)]
    loss_history = [[] for _ in range(cpus)]

    for step in range(100):
        ray.get([w.sgd_step.remote() for w in workers])
        futures_lists = ray.get([w.send_weights.remote() for w in workers])
        all_futures = [fut for sublist in futures_lists for fut in sublist]
        ray.get(all_futures)
        ray.get([w.update_weights.remote() for w in workers])

        # Record weights and loss for each worker at this step
        ws = ray.get([w.get_param.remote() for w in workers])
        losses = ray.get([w.compute_loss.remote() for w in workers])
        for i in range(cpus):
            weights_history[i].append(ws[i].numpy())
            loss_history[i].append(losses[i])

        if step % 10 == 0:
            print(f"Step {step} weights:")
            for i, w in enumerate(ws):
                print(f"  Worker {i}: {w.numpy()}")

    # After loop: Plot convergence of each worker's weights and loss
    import numpy as np
    steps = np.arange(100)

    plt.figure(figsize=(14, 6))

    # Plot weight components for each worker
    plt.subplot(1, 2, 1)
    for i in range(cpus):
        w_arr = np.array(weights_history[i])
        plt.plot(steps, w_arr[:, 0], label=f'Worker {i} weight 0')
        plt.plot(steps, w_arr[:, 1], label=f'Worker {i} weight 1', linestyle='--')
    plt.xlabel('Step')
    plt.ylabel('Weight values')
    plt.title('Weights convergence per worker')
    plt.legend(fontsize='small')

    # Plot loss per worker
    plt.subplot(1, 2, 2)
    for i in range(cpus):
        plt.plot(steps, loss_history[i], label=f'Worker {i}')
    plt.xlabel('Step')
    plt.ylabel('MSE Loss')
    plt.title('Loss convergence per worker')
    plt.legend(fontsize='small')

    plt.tight_layout()
    plt.savefig("decentralized_sgd_convergence.png")
    print("Plot saved to decentralized_sgd_convergence.png")
    
if __name__ == "__main__":
    main()
    ray.shutdown()
