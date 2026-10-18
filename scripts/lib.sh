# Shared helpers for pipeline scripts: env paths + a GPU lock so two pipelines never overlap.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Defaults are the local Windows envs; on the cluster set PY_GLIGEN / PY_MIGC / PY_EVAL (see scripts/cluster/).
G=${PY_GLIGEN:-/d/condamini/envs/mig_gligen/python.exe}
M=${PY_MIGC:-/d/condamini/envs/mig_migc/python.exe}
E=${PY_EVAL:-/d/condamini/envs/mig_eval/python.exe}
LOCK=outputs/.gpu_lock
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another pipeline holds $LOCK (pid $(cat $LOCK/pid 2>/dev/null)); exiting"; exit 3
fi
echo $$ > "$LOCK/pid"
trap 'rm -rf "$LOCK"' EXIT
export PYTHONUNBUFFERED=1
