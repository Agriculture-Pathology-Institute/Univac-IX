#!/usr/bin/env bash

# ==============================================================================
#   UNIVAC-IX CORE FABRIC SYSTEM LAUNCHER
#   Orchestrates Telemetry Streams, Watch Dogs, and Pre-Boot Validation Checks
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
# PHASE 1: MANDATORY PRE-BOOT VALIDATION CHECKS
# Automatically invokes test_fabric_health.py to check infrastructure bounds
# ------------------------------------------------------------------------------
echo "🔍 [PHASE 1] Launching Pre-Boot Integrity and Token Validation Checks..."

# Execute the diagnostic test suite using the standard workspace compiler
uv run python test_fabric_health.py
TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo -e "\n❌ [CRITICAL BOOT FAILURE]: Integrity validation checks returned errors (Code: $TEST_EXIT_CODE)."
    echo " ⚠️  [ABORTING STARTUP]: Rectify template leaks or file constraints before re-launching."
    echo "========================================================="
    exit $TEST_EXIT_CODE
fi

echo -e "\n✅ [INTEGRITY COMPLIANT]: Pre-boot diagnostics verified clear. Proceeding to runtime setup..."
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
# PHASE 2: LAUNCH BACKGROUND INFRASTRUCTURE RUNTIMES
# ------------------------------------------------------------------------------
echo "🚀 [PHASE 2] Spinning up parallel streaming runtime layers..."

# Step 1: Launch Telemetry Graph Watcher (Processes visio_mapping.csv changes)
echo " 📦 [1/3] Launching Real-Time Visio Hazard Monitor..."
uv run python monitor_visio_hazards.py > "$LOG_DIR/monitor_hazards.log" 2>&1 &
MONITOR_PID=$!
echo "      ↳ Active Thread Bound to PID: $MONITOR_PID"
sleep 1.0

# Step 2: Launch HTML Refresh Engine Register (Updates execution_report.html)
echo " 📦 [2/3] Launching Report Card Synchronizer Engine..."
uv run python watch_report_sync.py > "$LOG_DIR/report_sync.log" 2>&1 &
SYNC_PID=$!
echo "      ↳ Active Thread Bound to PID: $SYNC_PID"
sleep 1.0

# Step 3: Launch Core Data Multiplexer Stream (Modulates all 9 node profiles)
echo " 📦 [3/3] Launching Real-Time Telemetry Streaming Server..."
uv run python stream_all_nodes.py > "$LOG_DIR/telemetry_stream.log" 2>&1 &
STREAM_PID=$!
echo "      ↳ Active Thread Bound to PID: $STREAM_PID"

echo "---------------------------------------------------------"
echo "✅ RUNTIME BOOT SEQUENCE COMPLIANT."
echo "🌌 All 9 infrastructure segments operating concurrently in background slots."
echo "👉 Use 'tail -f $LOG_DIR/telemetry_stream.log' to inspect raw streams."
echo "👉 Press [Ctrl+C] at any time to dissolve the entire pipeline matrix."
echo "========================================================="

# Hold the bash pipeline execution loop open so the signal trap remains active
while true; do
    sleep 1
done
