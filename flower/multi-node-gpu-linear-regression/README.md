In order to run the following items, 

Part 1 - Superrlink (no gpu)
```
sudo apt-get update -y
sudo apt install python3-pip -y
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install -q flwr[simulation] flwr-datasets[vision] matplotlib
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
sudo apt install net-tools
```

Part 2.1 - GPU image
```
sudo apt-get update -y
sudo apt install python3-pip -y
aws s3 cp --recursive s3://ec2-linux-nvidia-drivers/latest/ .
aws s3 ls --recursive s3://ec2-linux-nvidia-drivers/
chmod +x NVIDIA-Linux-x86_64*.run
sudo /bin/sh ./NVIDIA-Linux-x86_64*.run
nvidia-smi -q | head
sudo apt install net-tools
```
Part 2.2 - Flower super node
```sudo apt install python3.10-venv
python3 -m venv doenv
source doenv/bin/activate
pip install -q flwr[simulation] flwr-datasets[vision] torch torchvision matplotlib
```
