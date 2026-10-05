#!/usr/bin/env bash

# ==============================================================================
#   UNIVAC-IX CORE FABRIC SYSTEM LAUNCHER
#   Orchestrates Parallel Telemetry Streams, Watch Dogs, and Alert Monitors
# ==============================================================================

# Global System Settings and Performance Overrides
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES="1"
export RAY_ENABLE_UV_RUN_RUNTIME_ENV="0"

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

echo "========================================================="
echo "⚙️  UNIVAC-IX: Booting Global Infrastructure Fabric Matrix"
echo "========================================================="
echo " 👉 Target Workspace: $(pwd)"
echo " 👉 Logging Directory: $LOG_DIR"
echo "---------------------------------------------------------"

# ------------------------------------------------------------------------------
# CLEANUP PARALLEL PROCESS TRAP ENGINE
# Ensures no loose Python processes hang in memory if the operator presses Ctrl+C
# ------------------------------------------------------------------------------
cleanup_fabric_matrix() {
    echo -e "\n\n⚠️  [SHUTDOWN SIGNAL CAPTURED]: Detaching infrastructure components safely..."
    
    echo " -> Terminating Telemetry Streaming Server (PID: $STREAM_PID)..."
    kill "$STREAM_PID" 2>/dev/null
    
    echo " -> Terminating Report Card Synchronizer (PID: $SYNC_PID)..."
    kill "$SYNC_PID" 2>/dev/null
    
    echo " -> Terminating Visio Hazard Monitor (PID: $MONITOR_PID)..."
    kill "$MONITOR_PID" 2>/dev/null

    echo "✅ Global Core Fabric standing down safely. All pipeline slots cleared."
    exit 0
}

# Bind our cleanup function to the SIGINT (Ctrl+C) and SIGTERM kill patterns
trap cleanup_fabric_matrix SIGINT SIGTERM

# ------------------------------------------------------------------------------
# STEP 1: LAUNCH TELEMETRY GRAPH WATCHER
# This module reads visio_mapping.csv alerts and handles console audio alarms
# ------------------------------------------------------------------------------
echo " 📦 [1/3] Spinning up Real-Time Visio Hazard Monitor..."
uv run python monitor_visio_hazards.py > "$LOG_DIR/monitor_hazards.log" 2>&1 &
MONITOR_PID=$!
echo "      ↳ Active Thread Bound to PID: $MONITOR_PID"
sleep 1.0

# ------------------------------------------------------------------------------
# STEP 2: LAUNCH HTML REFRESH ENGINE REGISTER
# This module updates execution_report.html on critical load alterations
# ------------------------------------------------------------------------------
echo " 📦 [2/3] Spinning up Report Card Synchronizer Engine..."
uv run python watch_report_sync.py > "$LOG_DIR/report_sync.log" 2>&1 &
SYNC_PID=$!
echo "      ↳ Active Thread Bound to PID: $SYNC_PID"
sleep 1.0

# ------------------------------------------------------------------------------
# STEP 3: LAUNCH CORE DATA MULTIPLEXER STREAM
# This module continuously updates changing matrix loads and variance indicators
# ------------------------------------------------------------------------------
echo " 📦 [3/3] Spinning up Real-Time Telemetry Streaming Server..."
uv run python stream_all_nodes.py > "$LOG_DIR/telemetry_stream.log" 2>&1 &
STREAM_PID=$!
echo "      ↳ Active Thread Bound to PID: $STREAM_PID"

echo "---------------------------------------------------------"
echo "✅ RUNTIME BOOT SEQUENCE COMPLIANT."
echo "🌌 Systems operating simultaneously in the background."
echo "👉 Use 'tail -f $LOG_DIR/telemetry_stream.log' to inspect raw streams."
echo "👉 Press [Ctrl+C] at any time to dissolve the entire pipeline matrix."
echo "========================================================="

# Hold the bash pipeline execution loop open so the signal trap remains active
while true; do
    sleep 1
done
