#!/usr/bin/env bash
# %% pinned runtime execution
set -euo pipefail

[[ "$CHECKER_REVISION" =~ ^[0-9a-f]{40}$ ]]
[[ "$SOURCE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
[[ "$RUNTIME_IMAGE" =~ ^[a-zA-Z0-9./:_-]+@sha256:[0-9a-f]{64}$ ]]
available_kib=$(awk '/MemAvailable:/ {print $2}' /proc/meminfo)
minimum_free_kib=5242880
if (( available_kib < minimum_free_kib )); then
    echo "Knowledge publication requires 5 GiB available memory."
    exit 1
fi
output="$RUNNER_TEMP/knowledge-output"
mkdir "$output"
arguments=(--revision "$SOURCE_COMMIT")
network=none
credential=()
if [[ "$OPERATION" == check ]]; then
    [[ "$BASE_COMMIT" =~ ^[0-9a-f]{40}$ ]]
    arguments+=(--base-revision "$BASE_COMMIT")
elif [[ "$OPERATION" == publish ]]; then
    network=bridge
    credential=(--env KNOWLEDGE_IDENTITY_TOKEN)
    arguments+=(--output /output/canonical)
elif [[ "$OPERATION" == consume ]]; then
    [[ "$ARTIFACT_IDENTIFIER" =~ ^[1-9][0-9]*$ ]]
    network=bridge
    credential=(--env KNOWLEDGE_IDENTITY_TOKEN)
    arguments+=(--artifact-id "$ARTIFACT_IDENTIFIER")
else
    echo "Unknown publication operation."
    exit 1
fi
docker pull "$RUNTIME_IMAGE"
docker run --rm --read-only --memory 4g --memory-swap 4g --pids-limit 256 \
    --cap-drop ALL --security-opt no-new-privileges --network "$network" \
    --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,noexec,nosuid,size=256m \
    --mount "type=bind,source=$GITHUB_WORKSPACE/checker,target=/checker,readonly" \
    --mount "type=bind,source=$GITHUB_WORKSPACE/proposal,target=/proposal,readonly" \
    --mount "type=bind,source=$output,target=/output" \
    --workdir /checker --env PYTHONDONTWRITEBYTECODE=1 \
    --env GIT_OPTIONAL_LOCKS=0 \
    --env PYTHONPATH=/checker/repository_knowledge/src:/checker/krrood/src \
    --env OPENBLAS_NUM_THREADS=1 --env OMP_NUM_THREADS=1 "${credential[@]}" \
    --entrypoint /opt/ros/cram-env/bin/python "$RUNTIME_IMAGE" \
    -m repository_knowledge publication --repository /proposal \
    --trusted-repository /checker --checker-revision "$CHECKER_REVISION" \
    "$OPERATION" "${arguments[@]}" > "$output/result.json"
if [[ "$OPERATION" == consume ]]; then
    echo "Verified artifact $ARTIFACT_IDENTIFIER for source $SOURCE_COMMIT."
else
    cat "$output/result.json"
fi
