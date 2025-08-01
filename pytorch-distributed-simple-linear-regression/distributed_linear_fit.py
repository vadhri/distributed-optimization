import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from datetime import timedelta

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    # Explicit timeout of 150 minutes
    dist.init_process_group("gloo", rank=rank, world_size=world_size, timeout=timedelta(seconds=9000))

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def train(rank, world_size, X_full, y_full, epochs=10000, lr=0.001, momentum=0.9):
    setup(rank, world_size)
    
    # Split data among processes
    N = X_full.shape[0]
    local_n = N // world_size
    start = rank * local_n
    end = start + local_n if rank != world_size - 1 else N

    X = X_full[start:end]
    y = y_full[start:end]

    # Initialize weights (same across all processes)
    torch.manual_seed(0)
    if rank == 0:
        w = torch.randn(2, 1, requires_grad=True)
        b = torch.randn(1, requires_grad=True)
    else:
        w = torch.zeros(2, 1, requires_grad=True)
        b = torch.zeros(1, requires_grad=True)
    
    # Broadcast initial weights
    dist.broadcast(w, src=0)
    dist.broadcast(b, src=0)

    # Initialize momentum buffers
    v_w = torch.zeros_like(w)
    v_b = torch.zeros_like(b)

    # Initialize convergence flag
    converged = torch.tensor([0], dtype=torch.int32)

    # Initialize loss history
    losses = []

    print(f"Process {rank} started.")
    
    try:
        for epoch in range(epochs):
            # Forward pass
            y_hat = X @ w + b
            loss = ((y_hat - y)**2).mean()

            # Save loss for visualization
            losses.append(loss.item())

            # Backward pass
            loss.backward()
            
            # Synchronize gradients and apply momentum
            with torch.no_grad():
                grad_w = w.grad.detach().clone()
                grad_b = b.grad.detach().clone()

                dist.all_reduce(grad_w, op=dist.ReduceOp.SUM)
                dist.all_reduce(grad_b, op=dist.ReduceOp.SUM)

                grad_w /= world_size
                grad_b /= world_size

                # Momentum update
                v_w = momentum * v_w + (1 - momentum) * grad_w
                v_b = momentum * v_b + (1 - momentum) * grad_b

                w -= lr * v_w
                b -= lr * v_b

                w.grad.zero_()
                b.grad.zero_()

            # Check for local convergence
            if loss.item() < 1e-4:
                converged[0] = 1
                print(f"Process {rank} converged at epoch {epoch+1}, Loss: {loss.item():.4f}")

            # Synchronize convergence status
            dist.all_reduce(converged, op=dist.ReduceOp.SUM)
            if converged.item() == world_size:  # All processes have converged
                print(f"Process {rank} exiting as all processes converged.")
                break

            # Log from all processes
            if rank == 0 and epoch % 100 == 0:
                print(f"Process {rank}, Epoch {epoch+1}, Loss: {loss.item():.6f}, w = {w.tolist()}, b = {b.item():.4f}")

        # Save losses to file
        torch.save(losses, f"losses_rank_{rank}.pt")

        #torch save the model parameters
        torch.save({'w': w, 'b': b}, f"model_rank_{rank}.pt")

    except RuntimeError as e:
        print(f"Process {rank} encountered error: {e}")
        raise
    finally:
        cleanup()

def main():
    world_size = 10
    print(f"Using {world_size} processes.")

    # Generate synthetic data
    N = 200
    torch.manual_seed(0)
    x1 = torch.linspace(-5, 5, N)
    x2 = torch.randn(N) * 2  # Independent random values for x2
    X = torch.stack((x1, x2), dim=1)  # shape (N, 2)
    
    # Normalize input features
    X = (X - X.mean(dim=0)) / X.std(dim=0)
    
    # Define true weights as 2D tensor
    true_w = torch.tensor([[2.0], [-3.0]])  # shape (2, 1)
    bias = 5.0
    y = X @ true_w + bias + torch.randn(N, 1) * 0.01  # shape (N, 1)

    # Ensure port is free
    os.environ["MASTER_PORT"] = "12355"
    try:
        mp.spawn(train, args=(world_size, X, y), nprocs=world_size, join=True)
    except Exception as e:
        print(f"Main process caught exception: {e}")
        raise

if __name__ == "__main__":
    main()
