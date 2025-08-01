import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def ring_all_reduce(tensor, rank, world_size):
    chunks = tensor.chunk(world_size)
    recv_buf = torch.zeros_like(chunks[0])

    # Reduce-scatter
    for i in range(world_size - 1):
        send_chunk = chunks[(rank - i) % world_size].clone()
        recv_rank = (rank + 1) % world_size
        send_rank = (rank - 1 + world_size) % world_size
        dist.send(send_chunk, dst=recv_rank)
        dist.recv(recv_buf, src=send_rank)
        chunks[(rank - i - 1) % world_size] += recv_buf

    # All-gather
    for i in range(world_size - 1):
        send_chunk = chunks[(rank - i - 1) % world_size].clone()
        recv_rank = (rank + 1) % world_size
        send_rank = (rank - 1 + world_size) % world_size
        dist.send(send_chunk, dst=recv_rank)
        dist.recv(recv_buf, src=send_rank)
        chunks[(rank - i - 2) % world_size] = recv_buf.clone()

    return torch.cat(chunks)

def train(rank, world_size, X_full, y_full, epochs=50, lr=0.01):
    setup(rank, world_size)

    # Split data among processes
    N = X_full.shape[0]
    local_n = N // world_size
    start = rank * local_n
    end = start + local_n if rank != world_size - 1 else N

    X = X_full[start:end]
    y = y_full[start:end]

    # Initialize weights
    w = torch.randn(2, requires_grad=True)

    for epoch in range(epochs):
        w.requires_grad_(True)

        # Forward pass: ŷ = X @ w
        y_hat = X @ w
        loss = ((y_hat - y)**2).mean()

        # Backward pass
        loss.backward()
        with torch.no_grad():
            grad = w.grad.detach().clone()
            # Ring all-reduce
            avg_grad = ring_all_reduce(grad, rank, world_size)
            avg_grad /= world_size
            # SGD step
            w -= lr * avg_grad
            w.grad.zero_()

        if rank == 0 and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, w = {w.tolist()}")

    cleanup()

def main():
    world_size = os.cpu_count()/4
    print(f"Using {world_size} processes.")

    # Generate synthetic data
    N = 20000
    torch.manual_seed(0)
    X = torch.randn(N, 2)
    y = 2 * X[:, 0] + 3 * X[:, 1] + 0.1 * torch.randn(N)

    mp.spawn(train, args=(world_size, X, y), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
