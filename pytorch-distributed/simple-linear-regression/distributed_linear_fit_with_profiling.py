import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta
import time
import psutil
import torch.profiler

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=9000))

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def train(rank, world_size, X_full, y_full, epochs=10000, lr=0.01, momentum=0.9):
    setup(rank, world_size)
    
    # Split data among processes
    N = X_full.shape[0]
    local_n = N // world_size
    start = rank * local_n
    end = start + local_n if rank != world_size - 1 else N
    X = X_full[start:end]
    y = y_full[start:end]

    # Initialize weights
    torch.manual_seed(0)
    if rank == 0:
        w = torch.randn(2, 1, requires_grad=True)
        b = torch.randn(1, requires_grad=True)
    else:
        w = torch.zeros(2, 1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
    
    dist.broadcast(w, src=0)
    dist.broadcast(b, src=0)

    v_w = torch.zeros_like(w)
    v_b = torch.zeros_like(b)
    converged = torch.tensor([0], dtype=torch.int32)
    losses = []
    comm_times = []  # Track communication time
    cpu_usages = []  # Track CPU usage

    # Initialize profiler
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"profiler_rank_{rank}"),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    )

    process = psutil.Process()  # For CPU usage
    profiler.start()
    
    try:
        for epoch in range(epochs):
            # Forward pass
            y_hat = X @ w + b
            loss = ((y_hat - y) ** 2).mean()
            losses.append(loss.item())

            # Backward pass
            loss.backward()
            
            # Synchronize gradients with timing
            with torch.no_grad():
                grad_w = w.grad.detach().clone()
                grad_b = b.grad.detach().clone()

                start_time = time.time()
                dist.all_reduce(grad_w, op=dist.ReduceOp.SUM)
                dist.all_reduce(grad_b, op=dist.ReduceOp.SUM)
                comm_time = time.time() - start_time
                comm_times.append(comm_time)

                grad_w /= world_size
                grad_b /= world_size

                v_w = momentum * v_w + (1 - momentum) * grad_w
                v_b = momentum * v_b + (1 - momentum) * grad_b
                w -= lr * v_w
                b -= lr * v_b
                w.grad.zero_()
                b.grad.zero_()

            # CPU usage
            cpu_usage = process.cpu_percent(interval=None)
            cpu_usages.append(cpu_usage)

            # Profiler step
            profiler.step()

            if loss.item() < 1e-3:
                converged[0] = 1
                print(f"Process {rank} converged at epoch {epoch+1}, Loss: {loss.item():.6f}")
            
            dist.all_reduce(converged, op=dist.ReduceOp.SUM)
            if converged.item() == world_size:
                print(f"Process {rank} exiting as all processes converged.")
                break

            if epoch % 100 == 0:
                print(f"Process {rank}, Epoch {epoch+1}, Loss: {loss.item():.6f}, "
                      f"Comm Time: {comm_time:.6f}s, CPU Usage: {cpu_usage:.2f}%")

        # Save losses, model, and stats
        torch.save(losses, f"losses_rank_{rank}.pt")
        torch.save({"w": w, "b": b}, f"model_rank_{rank}.pt")
        torch.save({"comm_times": comm_times, "cpu_usages": cpu_usages}, f"stats_rank_{rank}.pt")

    except RuntimeError as e:
        print(f"Process {rank} encountered error: {e}")
        raise
    finally:
        profiler.stop()
        cleanup()

def main():
    world_size = 10
    print(f"Using {world_size} processes")
    # Generate data
    N = 200
    torch.manual_seed(0)
    x1 = torch.linspace(-5, 5, N)
    x2 = torch.randn(N) * 2
    X = torch.stack((x1, x2), dim=1)
    X = (X - X.mean(dim=0)) / X.std(dim=0)  # Normalize
    true_w = torch.tensor([[2.0], [-3.0]])
    bias = 5.0
    y = X @ true_w + bias + torch.randn(N, 1) * 0.01

    mp.spawn(train, args=(world_size, X, y), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
