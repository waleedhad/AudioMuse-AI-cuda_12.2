name: Build, Test, and Push AudioMuse AI Docker Image

on:
  push:
    branches:
      - devel-librosa # Trigger only for 'devel-librosa' branch

jobs:
  build-and-test:
    runs-on: ${{ matrix.runner }} # Use runner based on matrix
    strategy:
      matrix:
        include:
          - platform: linux/amd64
            runner: ubuntu-latest
            arch_tag: amd64
          - platform: linux/arm64
            runner: ubuntu-latest-arm64
            arch_tag: arm64

    permissions:
      contents: read # Allow checkout to read repository contents
      packages: write # Allow pushing to GitHub Container Registry

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3 # Essential for multi-architecture builds

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Determine Docker image tags
        id: docker_tags
        run: |
          REPO_NAME_LOWER=$(echo "${{ github.repository }}" | tr '[:upper:]' '[:lower:]')
          ALL_TAGS=""

          if [[ "${GITHUB_REF}" == "refs/heads/devel-librosa" ]]; then
            DEV_LIBROSA_TAG="ghcr.io/$REPO_NAME_LOWER:devel-librosa"
            ALL_TAGS="$DEV_LIBROSA_TAG"
            echo "Building devel-librosa tag: $DEV_LIBROSA_TAG"

          elif [[ "${GITHUB_REF}" == refs/tags/v* ]]; then
            VERSION_TAG=$(echo "${GITHUB_REF}" | sed -e 's|refs/tags/v||g')
            VERSIONED_TAG="ghcr.io/$REPO_NAME_LOWER:${VERSION_TAG}"
            ALL_TAGS="$VERSIONED_TAG"
            echo "Building versioned tag: $VERSIONED_TAG"
          fi

          # Export the main tags for subsequent steps
          echo "docker_tags=$ALL_TAGS" >> "$GITHUB_OUTPUT"
          # Export a unique temporary tag for internal multi-architecture testing, specific to the current architecture
          echo "test_tag=ghcr.io/$REPO_NAME_LOWER:test-temp-${GITHUB_RUN_ID}-${{ matrix.arch_tag }}" >> "$GITHUB_OUTPUT"

      - name: Build and Push Docker image (for current architecture testing)
        id: docker_build_test
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile
          platforms: ${{ matrix.platform }} # Build only for the current matrix platform
          push: true # Push the temporary image to the registry for testing
          tags: ${{ steps.docker_tags.outputs.test_tag }}

      - name: Run Flask App Container Health Check (${{ matrix.arch_tag }})
        run: |
          echo "Starting temporary PostgreSQL container for Flask test (${{ matrix.arch_tag }})..."
          # No --platform needed here as the runner is native to the architecture
          docker run -d --name postgres-flask-test \
            -e POSTGRES_USER=testuser \
            -e POSTGRES_PASSWORD=testpass \
            -e POSTGRES_DB=testdb \
            postgres:latest

          echo "Waiting for PostgreSQL (Flask test ${{ matrix.arch_tag }}) to start..."
          for i in $(seq 1 30); do
            if docker exec postgres-flask-test pg_isready -U testuser -d testdb -q; then
              echo "PostgreSQL (Flask test ${{ matrix.arch_tag }}) is ready."
              break
            fi
            echo "Waiting for PostgreSQL (Flask test ${{ matrix.arch_tag }})... ($i/30)"
            sleep 2
            if [ $i -eq 30 ]; then
              echo "PostgreSQL (Flask test ${{ matrix.arch_tag }}) did not become ready in time."
              docker logs postgres-flask-test
              exit 1
            fi
          done

          echo "Starting Flask app container for health check (${{ matrix.arch_tag }})..."
          # Use a consistent port as runners are isolated
          docker run -d --name flask-test-app -p 8000:8000 \
            --link postgres-flask-test:postgres \
            -e SERVICE_TYPE=flask \
            -e POSTGRES_USER=testuser \
            -e POSTGRES_PASSWORD=testpass \
            -e POSTGRES_DB=testdb \
            -e POSTGRES_HOST=postgres \
            -e POSTGRES_PORT=5432 \
            ${{ steps.docker_tags.outputs.test_tag }}

          echo "Waiting for Flask app (${{ matrix.arch_tag }}) to start (max 60 seconds)..."
          for i in $(seq 1 60); do
            if curl -sf -I http://localhost:8000/api/last_task | grep -qE "HTTP/[12](.[01])? (2|3)[0-9]{2}"; then
              echo "Flask app (${{ matrix.arch_tag }}) is up and responsive!"
              exit 0
            fi
            echo "Waiting... ($i/60)"
            sleep 1
          done
          echo "Flask app (${{ matrix.arch_tag }}) did not start or respond with a successful status on /api/last_task within the expected time."
          docker logs flask-test-app
          exit 1

      - name: Run RQ Worker Container Health Check (${{ matrix.arch_tag }})
        run: |
          echo "Starting temporary Redis container for RQ worker (${{ matrix.arch_tag }})..."
          docker run -d --name redis-test redis:latest

          echo "Waiting for Redis (${{ matrix.arch_tag }}) to start..."
          for i in $(seq 1 30); do
            if docker exec redis-test redis-cli ping | grep -q PONG; then
              echo "Redis (${{ matrix.arch_tag }}) is ready."
              break
            fi
            echo "Waiting for Redis (${{ matrix.arch_tag }})... ($i/30)"
            sleep 1
            if [ $i -eq 30 ]; then
              echo "Redis (${{ matrix.arch_tag }}) did not become ready in time."
              docker logs redis-test
              exit 1
            fi
          done

          echo "Starting temporary PostgreSQL container for RQ worker test (${{ matrix.arch_tag }})..."
          docker run -d --name postgres-rq-test \
            -e POSTGRES_USER=testuser \
            -e POSTGRES_PASSWORD=testpass \
            -e POSTGRES_DB=testdb \
            postgres:latest

          echo "Waiting for PostgreSQL (RQ worker test ${{ matrix.arch_tag }}) to start..."
          for i in $(seq 1 30); do
            if docker exec postgres-rq-test pg_isready -U testuser -d testdb -q; then
              echo "PostgreSQL (RQ worker test ${{ matrix.arch_tag }}) is ready."
              break
            fi
            echo "Waiting for PostgreSQL (RQ worker test ${{ matrix.arch_tag }})... ($i/30)"
            sleep 2
            if [ $i -eq 30 ]; then
              echo "PostgreSQL (RQ worker test ${{ matrix.arch_tag }}) did not become ready in time."
              docker logs postgres-rq-test
              exit 1
            fi
          done

          echo "Starting RQ worker container for health check (${{ matrix.arch_tag }})..."
          docker run -d --name rq-test-worker \
            --link redis-test:redis --link postgres-rq-test:postgres \
            -e SERVICE_TYPE=worker \
            -e REDIS_URL=redis://redis:6379/0 \
            -e POSTGRES_USER=testuser \
            -e POSTGRES_PASSWORD=testpass \
            -e POSTGRES_DB=testdb \
            -e POSTGRES_HOST=postgres \
            -e POSTGRES_PORT=5432 \
            ${{ steps.docker_tags.outputs.test_tag }}

          echo "Waiting for RQ worker (${{ matrix.arch_tag }}) to start (max 90 seconds)..."
          for i in $(seq 1 30); do
            if docker ps -f name=rq-test-worker --format '{{.Status}}' | grep -q 'Up'; then
              sleep 2
              if docker logs rq-test-worker 2>&1 | grep -E "Listening on|RQ Worker [^ ]+ started"; then
                  echo "RQ worker container (${{ matrix.arch_tag }}) is running and listening."
                  exit 0
              fi
            fi
            echo "Waiting... ($i/30)"
            sleep 3
          done
          echo "RQ worker container (${{ matrix.arch_tag }}) did not start or become ready within the expected time."
          docker logs rq-test-worker
          exit 1

      - name: Clean up Docker containers and temporary image
        if: always() # Run this step even if previous steps fail
        run: |
          echo "Cleaning up temporary Docker containers and image..."
          docker rm -f flask-test-app || true
          docker rm -f postgres-flask-test || true
          docker rm -f rq-test-worker || true
          docker rm -f redis-test || true
          docker rm -f postgres-rq-test || true
          docker rmi ${{ steps.docker_tags.outputs.test_tag }} || true

  # This job will run only once after the build-and-test matrix is complete
  final-push:
    needs: build-and-test # Ensure this job runs after all matrix jobs
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Determine Docker image tags for final push
        id: docker_tags_final
        run: |
          REPO_NAME_LOWER=$(echo "${{ github.repository }}" | tr '[:upper:]' '[:lower:]')
          ALL_TAGS=""

          if [[ "${GITHUB_REF}" == "refs/heads/devel-librosa" ]]; then
            DEV_LIBROSA_TAG="ghcr.io/$REPO_NAME_LOWER:devel-librosa"
            ALL_TAGS="$DEV_LIBROSA_TAG"
            echo "Building devel-librosa tag: $DEV_LIBROSA_TAG"

          elif [[ "${GITHUB_REF}" == refs/tags/v* ]]; then
            VERSION_TAG=$(echo "${GITHUB_REF}" | sed -e 's|refs/tags/v||g')
            VERSIONED_TAG="ghcr.io/$REPO_NAME_LOWER:${VERSION_TAG}"
            ALL_TAGS="$VERSIONED_TAG"
            echo "Building versioned tag: $VERSIONED_TAG"
          fi
          echo "docker_tags=$ALL_TAGS" >> "$GITHUB_OUTPUT"

      - name: Build and Push AudioMuse AI Image (Final Multi-Arch Push)
        uses: docker/build-push-action@v5
        with:
          context: .
          file: ./Dockerfile
          platforms: linux/amd64,linux/arm64 # Build for both Intel and ARM architectures
          push: true
          tags: ${{ steps.docker_tags_final.outputs.docker_tags }}
