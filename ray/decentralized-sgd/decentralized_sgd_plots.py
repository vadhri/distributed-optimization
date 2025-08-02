import ray
import torch
import matplotlib.pyplot as plt
import os

ray.shutdown()
ray.init(include_dashboard=False, num_cpus=10)

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
        self.received_weights = [self.w for _ in neighbors]

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
        if len(self.neighbors) == 0:
            return True
        idx = sender_id % len(self.neighbors)
        self.received_weights[idx] = w
        return True

    def update_weights(self):
        all_ws = [self.w] + self.received_weights
        self.w = sum(all_ws) / len(all_ws)

    def get_param(self):
        return self.w

    def get_loss(self):
        y_hat = self.X @ self.w
        loss = torch.mean((y_hat - self.y) ** 2).item()
        return loss

def get_neighbors(workers, topology):
    N = len(workers)
    neighbors_list = []
    if topology == 'line':
        for i in range(N):
            if i < N-1:
                neighbors_list.append([workers[i+1]])
            else:
                neighbors_list.append([])
    elif topology == 'ring':
        for i in range(N):
            neighbors_list.append([workers[(i + 1) % N]])
    elif topology == 'fully_connected':
        for i in range(N):
            neighbors = [workers[j] for j in range(N) if j != i]
            neighbors_list.append(neighbors)
    else:
        raise ValueError("Unknown topology")
    return neighbors_list

def run_experiment(topology, seed, save_dir):
    torch.manual_seed(seed)
    N = 1000
    x = torch.linspace(-1, 1, N).unsqueeze(1)
    true_w = torch.tensor([2.0, -3.0])
    X = torch.cat([x, torch.ones_like(x)], dim=1)
    y = X @ true_w + 0.1 * torch.randn(N)

    cpus = min(10, int(ray.available_resources().get('CPU', 1)))
    parts = torch.chunk(X, cpus)
    labels = torch.chunk(y, cpus)

    workers = [Worker.remote(i, parts[i], labels[i]) for i in range(cpus)]

    neighbors_list = get_neighbors(workers, topology)

    for worker, neighbors in zip(workers, neighbors_list):
        ray.get(worker.set_neighbors.remote(neighbors))

    steps = 100
    weights_history = [[] for _ in range(cpus)]
    loss_history = [[] for _ in range(cpus)]

    for step in range(steps):
        ray.get([w.sgd_step.remote() for w in workers])
        futures_lists = ray.get([w.send_weights.remote() for w in workers])
        all_futures = [fut for sublist in futures_lists for fut in sublist]
        ray.get(all_futures)
        ray.get([w.update_weights.remote() for w in workers])

        ws = ray.get([w.get_param.remote() for w in workers])
        ls = ray.get([w.get_loss.remote() for w in workers])

        for i, (w, l) in enumerate(zip(ws, ls)):
            weights_history[i].append(w.numpy().copy())
            loss_history[i].append(l)

    # Convert weights_history to numpy arrays shape (steps, 2)
    weights_np = [torch.tensor(h).numpy() for h in weights_history]

    # Plot weights and losses
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Weight params plot
    for i in range(cpus):
        axes[0].plot([h[0] for h in weights_np[i]], label=f'Worker {i} param 0')
        axes[0].plot([h[1] for h in weights_np[i]], label=f'Worker {i} param 1', linestyle='dashed')
    axes[0].set_title(f'{topology.capitalize()} topology - Weight parameter convergence')
    axes[0].set_ylabel('Weights')
    axes[0].legend(fontsize='small', ncol=2, loc='upper right')

    # Loss plot
    for i in range(cpus):
        axes[1].plot(loss_history[i], label=f'Worker {i}')
    axes[1].set_title(f'{topology.capitalize()} topology - Local Loss convergence')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss')
    axes[1].legend(fontsize='small', ncol=2, loc='upper right')

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    filename = os.path.join(save_dir, f'{topology}_convergence.png')
    plt.savefig(filename)
    plt.close(fig)
    print(f"Saved plot to {filename}")

def main():
    save_dir = 'decentralized_sgd_plots'
    topologies = ['line', 'ring', 'fully_connected']

    for topology in topologies:
        run_experiment(topology, seed=42, save_dir=save_dir)

    ray.shutdown()

if __name__ == "__main__":
    main()


