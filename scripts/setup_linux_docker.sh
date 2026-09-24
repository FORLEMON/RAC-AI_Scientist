#!/usr/bin/env bash
# Install Docker Engine from Docker's official Ubuntu apt repository.
# Run as root inside the chosen WSL2 Ubuntu distribution.
set -euo pipefail
source /etc/os-release
if [[ "$ID" != ubuntu || "$(id -u)" != 0 ]]; then
  echo 'Run this script as root in Ubuntu.' >&2
  exit 1
fi
if command -v docker >/dev/null && docker compose version >/dev/null; then
  systemctl start docker
  docker version
  exit 0
fi
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends ca-certificates curl
install -m 0755 -d /etc/apt/keyrings
curl --fail --show-error --location --retry 3 https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${UBUNTU_CODENAME:-$VERSION_CODENAME}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF
apt-get update
apt-get install -y --no-install-recommends docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker
docker version
docker compose version
