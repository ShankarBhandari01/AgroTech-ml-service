#!/usr/bin/env bash
# Runs ON the VM. Points the compose stack's `ml` service at a new image and restarts just that
# service. The Kotlin backend's scripts/deploy.sh is the sibling of this file and the model for it.
#
# Everything uses sudo because OS Login creates a per-principal user (you SSH in as
# your-email_gmail_com, CI as sa_<id>) and none of them are in the docker group — but all OS Login
# admins get passwordless sudo. Registry auth goes through the VM's own metadata token rather than
# `gcloud auth configure-docker`, which would write to the calling user's ~/.docker/config.json
# while `sudo docker` reads root's.
#
# Usage: deploy-ml.sh <full-image-ref> <registry-host>
set -euo pipefail

IMAGE="${1:?usage: deploy-ml.sh <image> <registry-host>}"
REGISTRY="${2:?usage: deploy-ml.sh <image> <registry-host>}"

COMPOSE_FILE=/opt/agri/docker-compose.prod.yml
ENV_FILE=/opt/agri/.env

# Preflight, before anything is pulled or restarted, so a misconfigured VM costs a failed job
# rather than an outage.
for f in "$COMPOSE_FILE" "$ENV_FILE"; do
  [ -f "$f" ] || { echo "::error::$f not found — has the backend ever deployed to this VM?"; exit 1; }
done

# The image must exist before the running container is touched. A typo'd tag would otherwise fail
# at `compose pull` with the old container already stopped, turning a deploy into an outage.
TOKEN=$(curl -sf -H "Metadata-Flavor: Google" \
  "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')

echo "$TOKEN" | sudo docker login -u oauth2accesstoken --password-stdin "https://$REGISTRY"

# `docker compose` validates the WHOLE project even when a single service is named, and APP_IMAGE
# lives only in the backend deploy's environment — never in .env. Without this, every command below
# dies with "service app has neither an image nor a build context". Reading it back off the running
# container is what keeps this deploy from having an opinion about which backend is live.
APP_IMAGE="$(sudo docker ps --filter name=agri-app --format '{{.Image}}' | head -1)"
if [ -z "$APP_IMAGE" ]; then
  echo "::error::No running agri-app container to read APP_IMAGE from; refusing to guess."
  exit 1
fi
export APP_IMAGE

# ML_IMAGE is PERSISTED to .env, not just passed inline for this one command.
#
# That is the whole reason this script edits a file. The backend's deploy.sh runs
# `docker compose up -d` across every service with only APP_IMAGE overridden — so the `ml` service's
# image comes from .env. Pass the new tag inline and the container is correct until the next backend
# deploy silently reverts it to whatever .env still says, hours or days later.
sudo cp "$ENV_FILE" "$ENV_FILE.bak.$(date +%Y%m%d-%H%M%S)"
if sudo grep -q '^ML_IMAGE=' "$ENV_FILE"; then
  sudo sed -i "s|^ML_IMAGE=.*|ML_IMAGE=$IMAGE|" "$ENV_FILE"
else
  echo "ML_IMAGE=$IMAGE" | sudo tee -a "$ENV_FILE" >/dev/null
fi
echo "ML_IMAGE set to $IMAGE"

cd /opt/agri
sudo -E docker compose -f "$COMPOSE_FILE" pull ml
sudo -E docker compose -f "$COMPOSE_FILE" up -d ml

# Verify. Compose returns as soon as the container is STARTED, which says nothing about whether the
# service came up — a bad image or a missing model artifact both start fine and then fail. Exiting
# non-zero here is what makes a broken deploy a red job instead of a green one with a dead endpoint.
echo "Waiting for agri-ml to report healthy..."
for _ in $(seq 1 30); do
  status="$(sudo docker inspect --format '{{.State.Health.Status}}' agri-ml 2>/dev/null || echo missing)"
  case "$status" in
    healthy) echo "agri-ml is healthy on $IMAGE"; sudo docker image prune -f >/dev/null; exit 0 ;;
    unhealthy) echo "::error::agri-ml reported unhealthy"; sudo docker logs agri-ml --tail 50; exit 1 ;;
  esac
  sleep 5
done

echo "::error::agri-ml did not become healthy within 150s"
sudo docker logs agri-ml --tail 50
exit 1
