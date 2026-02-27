from typing import Dict, Any, List
from datetime import datetime
import dateutil.parser

class RulesEngine:
    """
    Phase 3: Neuro-Symbolic Logic Engine.
    Examines LLM-extracted metadata using deterministic Python rules.
    Outputs a consolidated report with Red Flags, Warnings, and Info.
    """ 
    
    def __init__(self):
        pass

    def check_dates(self, effective: str, expiration: str) -> List[Dict[str, str]]:
        flags = []
        if effective and expiration and effective.lower() != "none mentioned" and expiration.lower() != "none mentioned":
            try:
                eff_dt = dateutil.parser.parse(effective, fuzzy=True)
                exp_dt = dateutil.parser.parse(expiration, fuzzy=True)
                if eff_dt > exp_dt:
                    flags.append({
                        "level": "CRITICAL",
                        "field": "Dates",
                        "message": f"Effective Date ({effective}) is AFTER Expiration Date ({expiration})."
                    })
            except Exception:
                # Could not parse dates, ignore
                pass
        return flags

    def check_auto_renewal_trap(self, renewal: str, notice_period: str) -> List[Dict[str, str]]:
        flags = []
        has_renewal = renewal and renewal.lower() != "none mentioned"
        has_notice = notice_period and notice_period.lower() != "none mentioned"
        
        if has_renewal and not has_notice:
            flags.append({
                "level": "WARNING",
                "field": "Renewal",
                "message": "Contact has a renewal term but NO notice period to terminate specified. High risk of auto-renewal trap."
            })
        return flags

    def check_liability(self, uncapped: str) -> List[Dict[str, str]]:
        flags = []
        if uncapped and str(uncapped).lower() in ["yes", "true"]:
            flags.append({
                "level": "CRITICAL",
                "field": "Liability",
                "message": "UNCAPPED LIABILITY detected. This exposes the company to infinite risk."
            })
        return flags

    def analyze(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """Runs all rules over the extracted data JSON."""
        if "error" in extracted_data:
            return {"status": "FAILED", "error": extracted_data["error"], "flags": []}

        flags = []
        
        # Rule 1: Date Logic
        eff = str(extracted_data.get("Effective Date", ""))
        exp = str(extracted_data.get("Expiration Date", ""))
        flags.extend(self.check_dates(eff, exp))
        
        # Rule 2: Auto-renewal trap
        ren = str(extracted_data.get("Renewal Term", ""))
        notc = str(extracted_data.get("Notice Period To Terminate Renewal", ""))
        flags.extend(self.check_auto_renewal_trap(ren, notc))
        
        # Rule 3: Liability
        uncapped = str(extracted_data.get("Uncapped Liability", ""))
        flags.extend(self.check_liability(uncapped))

        # Rule 4: Exclusivity Logic
        grant = str(extracted_data.get("License Grant", "")).lower()
        excl = str(extracted_data.get("Exclusivity", "")).lower()
        if grant in ["yes", "true"] and excl in ["none mentioned", "no", "false"]:
            flags.append({
                "level": "INFO", 
                "field": "License", 
                "message": "License granted appears to be non-exclusive."
            })

        # Rule 5: Anti-Assignment
        anti = str(extracted_data.get("Anti-Assignment", "")).lower()
        if anti in ["yes", "true"]:
            flags.append({
                "level": "WARNING",
                "field": "Assignment",
                "message": "Anti-assignment clause is present. Consent needed to transfer agreement."
            })

        return {
            "status": "SUCCESS",
            "flags": flags,
            "raw_data": extracted_data
        }
