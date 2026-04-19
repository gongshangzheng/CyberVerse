.PHONY: proto setup test test-py test-go test-integration build inference server frontend docker-up docker-down lint clean

# Proto generation (Python + Go)
proto:
	./scripts/generate_proto.sh

# First-time setup (install Python deps before proto so grpc_tools is available)
setup:
	pip install -e ".[dev,inference]"
	$(MAKE) proto
	cd frontend && npm install

# Testing
test: test-py test-go

test-py:
	python -m pytest tests/unit -v

test-go:
	cd server && go test ./... -v

# 真实 FlashHead 出视频（需 CUDA + 本地 checkpoints，见 tests/integration）
test-integration:
	python -m pytest tests/integration/ -m integration -v -s

# Development servers
#   Reads avatar runtime GPU settings from cyberverse_config.yaml; auto-selects python vs torchrun.
#   Override with env vars for ad-hoc testing:
#     WORLD_SIZE=2 CUDA_VISIBLE_DEVICES=0,1 make inference
inference:
	@bash ./scripts/inference.sh

server:
	# Load .env and ensure Go version are correct before starting livekit-enabled server.
	@if [ -x /usr/lib/go-1.25/bin/go ]; then export PATH=/usr/lib/go-1.25/bin:$$PATH; fi; set -a; [ -f ./.env ] && . ./.env; set +a; cd server && go run -tags livekit ./cmd/cyberverse-server/ --config ../cyberverse_config.yaml

frontend:
	cd frontend && npm run dev

# Build
build-go:
	# Build with the "livekit" tag so LiveKit functionality is enabled in production too.
	@if [ -x /usr/lib/go-1.25/bin/go ]; then export PATH=/usr/lib/go-1.25/bin:$$PATH; fi; cd server && go build -tags livekit -o ../bin/cyberverse-server ./cmd/cyberverse-server/

# Docker
docker-up:
	cd infra && docker compose up --build

docker-down:
	cd infra && docker compose down

# Clean generated files
clean:
	rm -f inference/generated/*_pb2*.py
	rm -f server/internal/pb/*.go
	rm -rf bin/
