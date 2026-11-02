#!/bin/sh
# bmd_bridge.sh: run one USPEX relaxation step as a host SLURM job step.
#
# Design: docs/structure_search.md, section 4 ("launch bridge"); issue #66.
#
# USPEX 25 runs inside its Singularity image, which has no srun and no
# python3. USPEX calls this script as its commandExecutable,
#
#     sh bmd_bridge.sh <step>
#
# with the CalcFold as the working directory, once per relaxation step, and
# waits for it to exit. Several copies run at the same time. The script asks
# the host-side launcher (backend/structure_search/bridge/launcher.py) to run
# VASP in this CalcFold, waits for the answer and exits with VASP's exit
# status. It uses only POSIX sh and coreutils (date, od, tr, ln, mv, rm, cat,
# sleep) plus awk, all present in the image.
#
# Protocol (all paths below are inside $BMD_BRIDGE_SPOOL; the launcher
# creates the directories before USPEX starts):
#
#   1. Write the request to tmp/<id>.req, then publish it as
#      requests/<id>.req with "ln", which never overwrites an existing file,
#      so publishing is atomic and two requests cannot share a name.
#   2. The launcher claims a request by renaming it to claimed/<id>.req,
#      runs VASP, and writes done/<id>.done (atomically, by rename). The
#      done file holds "status=<n>".
#   3. This script exits with <n>.
#
#   If this script gives up (timeout, launcher gone, launcher stopped, or a
#   signal), it withdraws its request: it renames requests/<id>.req to
#   withdrawn/<id>.req. If the launcher has already claimed it, the rename
#   fails and the script writes cancel/<id> (holding "status=<n>", the code
#   this script exits with) instead, which makes the launcher stop that VASP
#   step. Either way no VASP step keeps running for a
#   relaxation USPEX has stopped waiting for.
#
# Environment (all set explicitly by the launch code; see master.py):
#
#   BMD_BRIDGE_SPOOL              spool directory (required, absolute path)
#   BMD_BRIDGE_TIMEOUT_SECONDS    give up after this many seconds; 0 means
#                                 never (required, integer)
#   BMD_BRIDGE_POLL_SECONDS       seconds between checks (default 2; may be
#                                 a decimal such as 0.5)
#   BMD_BRIDGE_HEARTBEAT_SECONDS  give up if the launcher's heartbeat is
#                                 older than this (default 300, integer)
#
# Exit status:
#
#   0-255  the status the launcher recorded for the VASP step. The launcher
#          itself records 121-124 for the cases below, so these values mean
#          the same whether this script or the launcher decided.
#   120    usage or configuration error (nothing was run)
#   121    launcher missing or not responding (heartbeat too old)
#   122    the launcher rejected the request (for example, the CalcFold is
#          outside the search directory)
#   123    the launcher stopped (end of job or SIGTERM) before VASP finished
#   124    timed out after BMD_BRIDGE_TIMEOUT_SECONDS
#   143    this script received SIGTERM, SIGINT or SIGHUP

set -u

EXIT_USAGE=120
EXIT_LAUNCHER_LOST=121
EXIT_STOPPED=123
EXIT_TIMEOUT=124
EXIT_SIGNAL=143

log() {
    printf 'bmd_bridge: %s\n' "$*" >&2
}

is_uint() {
    case $1 in
        '' | *[!0-9]*) return 1 ;;
    esac
    return 0
}

if [ "$#" -ne 1 ]; then
    log "usage: sh bmd_bridge.sh <step>"
    exit "$EXIT_USAGE"
fi
step=$1
if ! is_uint "$step" || [ "$step" -lt 1 ]; then
    log "step must be a positive integer, got '$step'"
    exit "$EXIT_USAGE"
fi

spool=${BMD_BRIDGE_SPOOL:-}
timeout_s=${BMD_BRIDGE_TIMEOUT_SECONDS:-}
poll=${BMD_BRIDGE_POLL_SECONDS:-2}
heartbeat_limit=${BMD_BRIDGE_HEARTBEAT_SECONDS:-300}

case $spool in
    /*) ;;
    *)
        log "BMD_BRIDGE_SPOOL must be an absolute path, got '$spool'"
        exit "$EXIT_USAGE"
        ;;
esac
if ! is_uint "$timeout_s"; then
    log "BMD_BRIDGE_TIMEOUT_SECONDS must be a non-negative integer, got '$timeout_s'"
    exit "$EXIT_USAGE"
fi
case $poll in
    '' | *[!0-9.]* | *.*.* | . )
        log "BMD_BRIDGE_POLL_SECONDS must be a positive number, got '$poll'"
        exit "$EXIT_USAGE"
        ;;
esac
if ! is_uint "$heartbeat_limit" || [ "$heartbeat_limit" -lt 1 ]; then
    log "BMD_BRIDGE_HEARTBEAT_SECONDS must be a positive integer, got '$heartbeat_limit'"
    exit "$EXIT_USAGE"
fi

for sub in tmp requests claimed done cancel withdrawn; do
    if [ ! -d "$spool/$sub" ]; then
        log "spool directory $spool/$sub does not exist; is the launcher running?"
        exit "$EXIT_LAUNCHER_LOST"
    fi
done

calcfold=$(pwd -P) || exit "$EXIT_USAGE"
newline='
'
case $calcfold in
    *"$newline"*)
        log "CalcFold path contains a newline"
        exit "$EXIT_USAGE"
        ;;
esac

start=$(date +%s)

# Publish the request under a unique ID: step, time, process ID and 48 random
# bits. "ln" refuses to overwrite, so an ID that is somehow already in use is
# detected and a new one drawn.
published=no
attempt=0
while [ "$attempt" -lt 5 ]; do
    attempt=$((attempt + 1))
    rand=$(od -An -N6 -tx1 /dev/urandom | tr -d ' \n')
    id="s${step}-${start}-$$-${rand}"
    tmp="$spool/tmp/$id.req"
    request="$spool/requests/$id.req"
    if [ -e "$spool/done/$id.done" ] || [ -e "$spool/claimed/$id.req" ]; then
        continue
    fi
    if ! printf 'id=%s\nstep=%s\ncalcfold=%s\ncreated=%s\n' \
        "$id" "$step" "$calcfold" "$start" > "$tmp"; then
        log "cannot write $tmp"
        exit "$EXIT_USAGE"
    fi
    if ln "$tmp" "$request" 2>/dev/null; then
        rm -f "$tmp"
        published=yes
        break
    fi
    rm -f "$tmp"
done
if [ "$published" != yes ]; then
    log "could not publish a request with a unique ID"
    exit "$EXIT_USAGE"
fi

done_file="$spool/done/$id.done"

# Read the status from the done file. Prints nothing if it is malformed.
done_status() {
    awk -F= '$1 == "status" { print $2; exit }' "$done_file" 2>/dev/null
}

exit_with_done_status() {
    status=$(done_status)
    if ! is_uint "$status" || [ "$status" -gt 255 ]; then
        log "malformed done file $done_file"
        exit "$EXIT_USAGE"
    fi
    exit "$status"
}

# Withdraw the request, or ask the launcher to stop the step if it has
# already claimed it. If the step finished in the meantime, exit with its
# real status instead of giving up.
give_up() {
    code=$1
    reason=$2
    trap - TERM INT HUP
    if mv "$request" "$spool/withdrawn/$id.req" 2>/dev/null; then
        :
    elif [ ! -f "$done_file" ]; then
        if printf 'status=%s\n' "$code" > "$spool/tmp/$id.cancel"; then
            mv "$spool/tmp/$id.cancel" "$spool/cancel/$id" 2>/dev/null
        fi
    fi
    if [ -f "$done_file" ]; then
        exit_with_done_status
    fi
    log "$reason (request $id, step $step, $calcfold)"
    exit "$code"
}

trap 'give_up "$EXIT_SIGNAL" "interrupted by a signal"' TERM INT HUP

while :; do
    if [ -f "$done_file" ]; then
        exit_with_done_status
    fi
    if [ -e "$spool/stopped" ]; then
        give_up "$EXIT_STOPPED" "the launcher has stopped"
    fi
    now=$(date +%s)
    if [ "$timeout_s" -gt 0 ] && [ $((now - start)) -ge "$timeout_s" ]; then
        give_up "$EXIT_TIMEOUT" "timed out after ${timeout_s} s"
    fi
    beat=$(cat "$spool/heartbeat" 2>/dev/null) || beat=0
    if ! is_uint "$beat" || [ "$beat" -lt "$start" ]; then
        beat=$start
    fi
    if [ $((now - beat)) -ge "$heartbeat_limit" ]; then
        give_up "$EXIT_LAUNCHER_LOST" "no launcher heartbeat for ${heartbeat_limit} s"
    fi
    sleep "$poll"
done
