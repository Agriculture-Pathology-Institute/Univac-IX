/**
 * Univac-IX Sovereign Node Validation Suite
 * Manages structural tracking metrics across Node JSON payloads
 */

class AgriNodeValidator {
    constructor(nodeId, defaultCapacity = 100) {
        this.nodeId = nodeId;
        this.capacity = defaultCapacity;
    }

    /**
     * Parses and checks the functional state of incoming node telemetry maps
     * @param {Object} telemetry - The input metrics object containing sensor evaluations
     */
    validateTelemetry(telemetry) {
        const { moisture, nutrients, status } = telemetry;

        // Ensure variables sit inside logical boundaries
        if (moisture === undefined || nutrients === undefined) {
            throw new Error(`[Node ${this.nodeId}] Validation Failure: Incomplete sensory payload.`);
        }

        let dynamicMultiplier = 1.0;
        let requiresIntervention = false;

        if (moisture < 40.0 || nutrients < 50.0) {
            requiresIntervention = true;
            dynamicMultiplier = 0.75;
        }

        return {
            nodeId: this.nodeId,
            nominalUptime: status === 'ACTIVE',
            projectedYield: Math.round(this.capacity * dynamicMultiplier),
            triggerSystemOverride: requiresIntervention
        };
    }
}

// Operational Sandbox Execution
const telemetryPayload = { moisture: 35.5, nutrients: 72.0, status: 'ACTIVE' };
const validator = new AgriNodeValidator('Node-IX-01', 250);
const report = validator.validateTelemetry(telemetryPayload);

console.log(`Execution Report for ${report.nodeId}:`);
console.log(`- Projected Yield Capacity: ${report.projectedYield} units`);
console.log(`- Trigger Reverse-Injection Recovery: ${report.triggerSystemOverride}`);
