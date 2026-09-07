from typing import List, Dict, Any

class MaintenanceReportGenerator:
    """
    Natural Language Maintenance Report & Action Item Generator.
    Translates numeric ML outputs (SHAP drivers, RUL, failure probability) into
    structurally framed engineering notes for plant reliability operators.
    """
    def __init__(self):
        pass

    def generate_report(
        self, 
        unit_id: int, 
        cycle: int, 
        failure_prob: float, 
        predicted_rul: float, 
        anomaly_score: float, 
        top_drivers: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        
        # Determine operational urgency
        if failure_prob >= 0.80 or predicted_rul <= 15:
            urgency = "CRITICAL — IMMEDIATE INSPECTION REQUIRED"
        elif failure_prob >= 0.50 or predicted_rul <= 35:
            urgency = "HIGH — SCHEDULE MAINTENANCE THIS WEEK"
        elif failure_prob >= 0.25 or predicted_rul <= 60:
            urgency = "MEDIUM — ELEVATED MONITORING"
        else:
            urgency = "LOW — NORMAL OPERATING STATE"

        # Sensor translation map for human readability
        sensor_names = {
            "sensor_2": "LPC Outlet Temperature (T24)",
            "sensor_3": "HPC Outlet Temperature (T30)",
            "sensor_4": "LPT Outlet Temperature (T50)",
            "sensor_7": "HPC Outlet Pressure (P30)",
            "sensor_8": "Physical Fan Speed",
            "sensor_9": "Physical Core Speed",
            "sensor_11": "HPC Static Pressure",
            "sensor_12": "Fuel Flow Ratio",
            "sensor_14": "Corrected Fan Speed",
            "sensor_15": "Bypass Ratio",
            "sensor_17": "Bleed Enthalpy",
            "sensor_20": "HPT Coolant Bleed",
            "sensor_21": "LPT Coolant Bleed",
            "ratio_temp_press_2_7": "Thermal-to-Pressure Ratio (T24/P30)",
            "ratio_temp_3_4": "HPC/LPT Temp Ratio"
        }
        
        driver_descriptions = []
        action_items = []
        
        for d in top_drivers:
            feat = d.get("feature", "")
            base_feat = feat.split("_roll_")[0].split("_dev_")[0]
            clean_name = sensor_names.get(base_feat, feat)
            val = d.get("value", 0.0)
            impact = d.get("impact", "")
            
            if "roll_mean" in feat or "dev_baseline" in feat or base_feat in sensor_names:
                driver_descriptions.append(
                    f"• {clean_name} (reading: {val:.2f}) is driving elevated degradation risk (SHAP impact: {d.get('shap_value', 0.0):+.3f})."
                )
                
            # Add specific maintenance recommendations based on physical subsystem
            if "sensor_2" in feat or "sensor_3" in feat or "sensor_4" in feat:
                action_items.append("Inspect thermal insulation, combustor liners, and turbine blade cooling passages for thermal degradation.")
            elif "sensor_7" in feat or "sensor_11" in feat or "sensor_12" in feat:
                action_items.append("Inspect compressor seals, bleed valves, and fuel metering unit for pressure leakages or flow restrictions.")
            elif "sensor_8" in feat or "sensor_9" in feat or "sensor_14" in feat:
                action_items.append("Perform dynamic vibration analysis and inspect rotor bearings and shaft alignment for mechanical imbalance.")
            elif "sensor_15" in feat:
                action_items.append("Check bypass duct seals and fan blade pitch mechanism for aerodynamic efficiency loss.")
                
        if not action_items:
            action_items.append("Routine condition monitoring — verify sensor calibration during standard maintenance window.")

        # Remove duplicate action items while preserving order
        dedup_actions = list(dict.fromkeys(action_items))
        
        summary = (
            f"Asset #{unit_id} operating at Cycle {cycle} exhibits a {failure_prob:.1%} probability of failure "
            f"within the next horizon window, with estimated Remaining Useful Life (RUL) of {predicted_rul:.0f} cycles. "
            f"Anomaly Detector MSE score is {anomaly_score:.4f}."
        )
        
        root_cause_explanation = (
            f"Root cause attribution via SHAP value decomposition indicates primary physical drivers:\n" +
            "\n".join(driver_descriptions)
        )
        
        return {
            "unit_id": unit_id,
            "urgency": urgency,
            "maintenance_summary": summary,
            "action_items": dedup_actions,
            "root_cause_explanation": root_cause_explanation
        }
