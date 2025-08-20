provider "aws" {
  region = "us-east-1"
}

######################
# VPC + Networking
######################
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
}

resource "aws_subnet" "main" {
  vpc_id                  = aws_vpc.main.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "us-east-1b"
  map_public_ip_on_launch = true
}

resource "aws_internet_gateway" "gw" {
  vpc_id = aws_vpc.main.id
}

resource "aws_route_table" "rt" {
  vpc_id = aws_vpc.main.id
}

resource "aws_route" "default" {
  route_table_id         = aws_route_table.rt.id
  destination_cidr_block = "0.0.0.0/0"
  gateway_id             = aws_internet_gateway.gw.id
}

resource "aws_route_table_association" "a" {
  subnet_id      = aws_subnet.main.id
  route_table_id = aws_route_table.rt.id
}

######################
# Security Group
######################
resource "aws_security_group" "main" {
  vpc_id = aws_vpc.main.id
  name   = "flower-sg"

  # Allow SSH from outside
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow 9090-9099 only within VPC
  ingress {
    from_port   = 9090
    to_port     = 9099
    protocol    = "tcp"
    cidr_blocks = [aws_vpc.main.cidr_block]
  }

  # Egress: allow all
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

######################
# SSH Keys
######################
resource "tls_private_key" "ssh_keys" {
  count     = 2
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "keys" {
  count      = 2
  key_name   = "flower-key-${count.index + 1}"
  public_key = tls_private_key.ssh_keys[count.index].public_key_openssh
}

resource "local_file" "pem_files" {
  count    = 2
  filename = "${path.module}/flower-key-${count.index + 1}.pem"
  content  = tls_private_key.ssh_keys[count.index].private_key_pem
  file_permission = "0400"
}

######################
# EC2 Instances
######################
# 2 Spot Instances (g6g.large)
resource "aws_instance" "spot_instances" {
  count                = 1
  ami                  = "ami-06d63a11d8eaa8195" # Ubuntu 22.04 LTS ARM64 for us-east-1
  instance_type        = "g6f.large"
  subnet_id            = aws_subnet.main.id
  vpc_security_group_ids = [aws_security_group.main.id]
  key_name             = aws_key_pair.keys[count.index].key_name

  instance_market_options {
    market_type = "spot"
  }

  root_block_device {
    volume_size = 75        
    volume_type = "gp3"
    delete_on_termination = true
  }

  tags = {
    Name = "flower-spot-${count.index + 1}"
  }
}

# 1 On-Demand Instance (t3.micro)
resource "aws_instance" "ondemand_instance" {
  ami                  = "ami-053b0d53c279acc90" # Ubuntu 22.04 LTS x86_64 for us-east-1
  instance_type        = "t3.micro"
  subnet_id            = aws_subnet.main.id
  vpc_security_group_ids = [aws_security_group.main.id]
  key_name             = aws_key_pair.keys[1].key_name # third key

  root_block_device {
    volume_size = 20        
    volume_type = "gp3"
    delete_on_termination = true
  }

  tags = {
    Name = "flower-ondemand"
  }
}

######################
# Elastic IPs
######################
resource "aws_eip" "spot_ips" {
  count    = 1
  instance = aws_instance.spot_instances[count.index].id
}

resource "aws_eip" "ondemand_ip" {
  instance = aws_instance.ondemand_instance.id
}

######################
# Outputs
######################
output "spot_instance_ips" {
  value = aws_eip.spot_ips[*].public_ip
}

output "ondemand_instance_ip" {
  value = aws_eip.ondemand_ip.public_ip
}

output "pem_files" {
  value = local_file.pem_files[*].filename
}
