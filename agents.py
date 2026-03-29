#!/usr/bin/env python3
"""
MediAgent-Retina: AI Agent Modules
5 Specialized AI Agents for Retinal Disease Analysis
"""

import datetime
import random

# ════════════════════════════════════════════════════════════════════
# DISEASE KNOWLEDGE BASE (SHARED BY ALL AGENTS)
# ════════════════════════════════════════════════════════════════════

DISEASE_KNOWLEDGE_BASE = {
    "Normal": {
        "description": "The retina appears healthy with no detectable abnormalities. The optic disc, macula, and retinal vasculature show normal morphology and coloration.",
        "severity": "None",
        "urgency": "Routine",
        "color": "#4CAF50",
        "symptoms": [
            "No visual symptoms expected",
            "Maintain regular eye checkups for continued health"
        ],
        "risk_factors": [
            "Age over 40 (general screening recommended)",
            "Family history of eye disease",
            "Diabetes or hypertension",
            "Prolonged screen time"
        ],
        "treatment": [
            "No treatment needed",
            "Annual comprehensive eye exam recommended",
            "Maintain healthy lifestyle for long-term eye health"
        ],
        "precautions": [
            "Continue regular eye examinations every 1-2 years",
            "Wear UV-protective sunglasses outdoors",
            "Maintain a balanced diet rich in leafy greens and omega-3",
            "Stay hydrated and get adequate sleep",
            "Follow the 20-20-20 rule for screen use",
            "Exercise regularly to maintain good circulation"
        ],
        "demographics": {
            "common_age": "All ages",
            "gender_ratio": "Equal",
            "prevalence": "Majority of population",
            "geographic": "Universal"
        },
        "foods": [
            "Leafy greens (spinach, kale)",
            "Fish rich in omega-3 (salmon, tuna)",
            "Citrus fruits and berries",
            "Nuts and seeds",
            "Carrots and sweet potatoes",
            "Eggs (lutein and zeaxanthin)"
        ]
    },

    "Diabetic Retinopathy": {
        "description": "Diabetic retinopathy is a serious diabetes complication affecting retinal blood vessels. High blood sugar damages tiny vessels, causing leakage, bleeding, and abnormal new vessel growth, leading to progressive vision loss if untreated.",
        "severity": "High",
        "urgency": "URGENT",
        "color": "#f44336",
        "symptoms": [
            "Blurred or fluctuating vision",
            "Dark spots or floaters",
            "Difficulty perceiving colors",
            "Dark or empty areas in vision",
            "Progressive vision loss",
            "Difficulty with night vision"
        ],
        "risk_factors": [
            "Poorly controlled blood sugar (HbA1c > 7%)",
            "Long duration of diabetes (>10 years)",
            "Hypertension",
            "High cholesterol",
            "Pregnancy with diabetes",
            "Smoking",
            "Kidney disease (nephropathy)"
        ],
        "treatment": [
            "Strict blood sugar control (HbA1c < 7%)",
            "Anti-VEGF injections (Avastin, Lucentis, Eylea)",
            "Laser photocoagulation therapy",
            "Vitrectomy surgery for advanced cases",
            "Blood pressure and cholesterol management",
            "Regular monitoring every 3-6 months"
        ],
        "precautions": [
            "Monitor blood sugar levels daily - maintain HbA1c below 7%",
            "Control blood pressure (target < 130/80 mmHg)",
            "Take all diabetes medications as prescribed",
            "Schedule eye exams every 3-6 months",
            "Report any sudden vision changes immediately",
            "Avoid smoking and limit alcohol consumption",
            "Exercise regularly (30 min/day, 5 days/week)",
            "Monitor for signs of kidney disease",
            "Carry medical ID indicating diabetic status",
            "Keep emergency ophthalmology contact available"
        ],
        "demographics": {
            "common_age": "30-70 years",
            "gender_ratio": "Slightly higher in males",
            "prevalence": "~35% of diabetic patients worldwide",
            "geographic": "Higher in South Asia, Middle East, Latin America"
        },
        "foods": [
            "Low glycemic index foods (whole grains, legumes)",
            "Non-starchy vegetables (broccoli, spinach, peppers)",
            "Fatty fish (salmon, sardines, mackerel)",
            "Nuts and seeds (almonds, walnuts, flaxseeds)",
            "Berries (blueberries, strawberries)",
            "AVOID: refined sugars, white bread, sugary drinks"
        ]
    },

    "ARMD": {
        "description": "Age-Related Macular Degeneration (ARMD/AMD) is a progressive disease affecting the macula, the central part of the retina responsible for sharp detailed vision. It is the leading cause of irreversible vision loss in people over 50.",
        "severity": "Moderate to Severe",
        "urgency": "HIGH",
        "color": "#FF5722",
        "symptoms": [
            "Gradual loss of central vision",
            "Straight lines appear wavy or distorted (metamorphopsia)",
            "Difficulty recognizing faces",
            "Need for brighter light when reading",
            "Blurry or blind spot in central vision",
            "Colors appear less vivid"
        ],
        "risk_factors": [
            "Age over 50 (risk increases significantly after 60)",
            "Family history of AMD",
            "Smoking (2-3x increased risk)",
            "Caucasian ethnicity",
            "Obesity",
            "Cardiovascular disease",
            "Prolonged UV light exposure"
        ],
        "treatment": [
            "Anti-VEGF injections for wet AMD (Lucentis, Eylea)",
            "AREDS2 supplements (Vitamins C, E, zinc, lutein, zeaxanthin)",
            "Photodynamic therapy for certain wet AMD cases",
            "Low vision aids and rehabilitation",
            "Lifestyle modifications (smoking cessation, diet)",
            "Monitoring with Amsler grid at home daily"
        ],
        "precautions": [
            "Use Amsler grid daily to monitor for vision changes",
            "Wear UV-protective sunglasses with wide brims outdoors",
            "Stop smoking immediately - single most modifiable risk factor",
            "Take AREDS2 supplements as recommended by your doctor",
            "Maintain healthy blood pressure and cholesterol",
            "Eat a diet rich in lutein and zeaxanthin",
            "Schedule regular eye exams every 6-12 months",
            "Use adequate lighting for reading and close work",
            "Consider genetic testing if family history exists",
            "Report any sudden vision changes immediately"
        ],
        "demographics": {
            "common_age": "50-85+ years",
            "gender_ratio": "Slightly higher in females",
            "prevalence": "~11% of people over 65 globally",
            "geographic": "Higher in European and North American populations"
        },
        "foods": [
            "Dark leafy greens (kale, spinach, collard greens - lutein rich)",
            "Orange and yellow peppers (zeaxanthin)",
            "Salmon, tuna, sardines (omega-3 fatty acids)",
            "Eggs (lutein and zeaxanthin in yolks)",
            "Nuts (especially almonds and walnuts)",
            "AVOID: high-fat processed foods, excessive red meat"
        ]
    },

    "Media Haze": {
        "description": "Media haze refers to loss of transparency in the ocular media (cornea, aqueous humor, lens, or vitreous humor). This opacity obscures the view of the retina and indicates underlying pathology such as cataracts, corneal disease, or vitreous hemorrhage.",
        "severity": "Mild to Moderate",
        "urgency": "Moderate",
        "color": "#FF9800",
        "symptoms": [
            "Blurred or hazy vision",
            "Glare sensitivity especially at night",
            "Difficulty seeing in low light",
            "Faded or washed-out colors",
            "Halos around lights",
            "Progressive visual decline"
        ],
        "risk_factors": [
            "Cataracts (most common cause)",
            "Corneal edema or scarring",
            "Vitreous hemorrhage",
            "Uveitis or intraocular inflammation",
            "Post-surgical complications",
            "Age over 60",
            "Diabetes mellitus"
        ],
        "treatment": [
            "Treat underlying cause (cataract surgery, anti-inflammatory therapy)",
            "Corneal transplant if corneal opacity is significant",
            "Vitrectomy for vitreous opacity",
            "Anti-inflammatory medications for uveitis",
            "Monitoring for progression",
            "Corrective lenses as interim measure"
        ],
        "precautions": [
            "Seek prompt evaluation to determine the cause of haze",
            "Use prescribed anti-inflammatory eye drops as directed",
            "Protect eyes from UV exposure and trauma",
            "Avoid rubbing eyes vigorously",
            "Report any sudden worsening of vision",
            "Follow up regularly with ophthalmologist",
            "Manage underlying conditions (diabetes, hypertension)",
            "Use protective eyewear during risky activities"
        ],
        "demographics": {
            "common_age": "40-80 years (varies by cause)",
            "gender_ratio": "Equal",
            "prevalence": "Cataracts affect ~50% of people over 65",
            "geographic": "Higher in developing regions with limited eye care"
        },
        "foods": [
            "Vitamin C-rich foods (oranges, strawberries, bell peppers)",
            "Vitamin E sources (sunflower seeds, almonds)",
            "Omega-3 fatty acids (fish, flaxseeds)",
            "Colorful vegetables (carrots, sweet potatoes)",
            "Green tea (antioxidants)",
            "AVOID: excessive alcohol, high-sugar foods"
        ]
    },

    "Optic Disc Disease": {
        "description": "Optic disc diseases encompass conditions affecting the optic nerve head, including glaucoma, optic neuritis, papilledema, and optic atrophy. These can cause irreversible vision loss through damage to the optic nerve fibers.",
        "severity": "Moderate to Severe",
        "urgency": "HIGH",
        "color": "#E91E63",
        "symptoms": [
            "Gradual peripheral vision loss (tunnel vision)",
            "Eye pain or pressure sensation",
            "Headaches (especially with papilledema)",
            "Sudden vision loss in one eye (optic neuritis)",
            "Difficulty with contrast sensitivity",
            "Color vision changes"
        ],
        "risk_factors": [
            "Elevated intraocular pressure",
            "Family history of glaucoma",
            "Age over 40",
            "African or Hispanic ancestry",
            "High myopia",
            "Thin central cornea",
            "Autoimmune diseases (for optic neuritis)",
            "Intracranial hypertension (for papilledema)"
        ],
        "treatment": [
            "IOP-lowering eye drops (prostaglandins, beta-blockers)",
            "Laser trabeculoplasty or iridotomy",
            "Surgical intervention (trabeculectomy, tube shunts)",
            "Corticosteroids for optic neuritis",
            "Treatment of underlying cause for papilledema",
            "Neuroprotective therapies (emerging)",
            "Regular visual field monitoring"
        ],
        "precautions": [
            "Use prescribed eye drops at the SAME time every day",
            "Never skip or discontinue medications without doctor approval",
            "Have intraocular pressure checked every 3-6 months",
            "Perform periodic visual field tests",
            "Report any sudden vision changes or eye pain immediately",
            "Avoid activities that significantly increase eye pressure",
            "Inform all doctors about your optic disc condition",
            "Wear protective eyewear during sports",
            "Get adequate sleep - avoid sleeping face-down",
            "Limit caffeine intake to moderate levels"
        ],
        "demographics": {
            "common_age": "40-80 years (glaucoma); 20-50 (optic neuritis)",
            "gender_ratio": "Glaucoma slightly higher in males; optic neuritis higher in females",
            "prevalence": "Glaucoma affects ~3.5% of people over 40 globally",
            "geographic": "Higher glaucoma prevalence in African populations"
        },
        "foods": [
            "Leafy greens (high in nitrates - may improve optic nerve blood flow)",
            "Bilberries and blueberries (anthocyanins)",
            "Fish rich in omega-3 fatty acids",
            "Nuts and seeds (Vitamin E)",
            "Citrus fruits (Vitamin C)",
            "AVOID: excessive caffeine, very salty foods"
        ]
    },

    "Retinal Vascular Disease": {
        "description": "Retinal vascular diseases affect blood vessels of the retina, including retinal vein occlusion (RVO), retinal artery occlusion (RAO), and hypertensive retinopathy. These can cause sudden or gradual vision loss and are closely linked to cardiovascular health.",
        "severity": "Severe",
        "urgency": "URGENT",
        "color": "#9C27B0",
        "symptoms": [
            "Sudden painless vision loss (artery occlusion)",
            "Gradual vision blurring (vein occlusion)",
            "Distorted or dark areas in vision",
            "Floaters",
            "Visual field defects",
            "Metamorphopsia (distortion of straight lines)"
        ],
        "risk_factors": [
            "Hypertension (most common risk factor)",
            "Diabetes mellitus",
            "Atherosclerosis",
            "High cholesterol",
            "Smoking",
            "Blood clotting disorders",
            "Age over 50",
            "Glaucoma",
            "Oral contraceptive use"
        ],
        "treatment": [
            "Anti-VEGF injections for macular edema",
            "Intravitreal corticosteroid implants",
            "Laser photocoagulation for ischemic areas",
            "Aggressive cardiovascular risk factor management",
            "Anticoagulation therapy (selected cases)",
            "Pan-retinal photocoagulation for neovascularization",
            "Emergency treatment for central retinal artery occlusion"
        ],
        "precautions": [
            "Control blood pressure - this is CRITICAL (target < 130/80 mmHg)",
            "Take cardiovascular medications exactly as prescribed",
            "Monitor blood sugar if diabetic",
            "Stop smoking immediately",
            "Manage cholesterol levels (target LDL < 100 mg/dL)",
            "Seek EMERGENCY care for sudden vision loss",
            "Have regular cardiovascular checkups",
            "Stay physically active (30 min moderate exercise daily)",
            "Monitor for symptoms in the other eye",
            "Stay well hydrated throughout the day"
        ],
        "demographics": {
            "common_age": "50-75 years",
            "gender_ratio": "Slightly higher in males",
            "prevalence": "RVO affects ~1-2% of people over 40",
            "geographic": "Higher in populations with high cardiovascular disease burden"
        },
        "foods": [
            "Heart-healthy foods (olive oil, avocados)",
            "Fatty fish (salmon, mackerel - omega-3)",
            "Whole grains (oats, quinoa, brown rice)",
            "Low-sodium foods (fresh vegetables, herbs for seasoning)",
            "Garlic and onions (may improve circulation)",
            "AVOID: excessive salt, processed meats, trans fats"
        ]
    },

    "Myopia": {
        "description": "Pathological myopia is a severe form of nearsightedness where the eye elongates excessively, causing structural retinal changes including thinning, lacquer cracks, myopic macular degeneration, and increased risk of retinal detachment.",
        "severity": "Mild to Moderate",
        "urgency": "Moderate",
        "color": "#2196F3",
        "symptoms": [
            "Severely blurred distance vision",
            "Floaters or flashes of light",
            "Difficulty seeing at night",
            "Eye strain and headaches",
            "Progressive vision deterioration",
            "Peripheral vision loss (if complications develop)"
        ],
        "risk_factors": [
            "Genetic predisposition (parental myopia)",
            "Excessive near work (reading, screens)",
            "Limited outdoor time during childhood",
            "East Asian ethnicity (higher prevalence)",
            "Higher education level (correlated)",
            "Early onset of myopia in childhood"
        ],
        "treatment": [
            "Corrective lenses (glasses or contact lenses)",
            "Atropine eye drops (low-dose, myopia control in children)",
            "Orthokeratology (overnight contact lenses)",
            "Refractive surgery (LASIK, PRK - for stable myopia)",
            "Regular monitoring for retinal complications",
            "Surgical repair if retinal detachment occurs",
            "Anti-VEGF for myopic choroidal neovascularization"
        ],
        "precautions": [
            "Follow the 20-20-20 rule (every 20 min, look 20 feet away, 20 seconds)",
            "Ensure children spend at least 2 hours outdoors daily",
            "Use proper lighting when reading or working",
            "Have annual dilated eye exams to check retinal health",
            "Report any flashes, floaters, or curtain-like vision loss immediately",
            "Avoid contact sports without protective eyewear",
            "Consider myopia control treatments for children",
            "Maintain proper working distance from screens (arms length)",
            "Take regular breaks during prolonged near work",
            "Know the warning signs of retinal detachment"
        ],
        "demographics": {
            "common_age": "Onset in childhood/teens; complications in 30-60 years",
            "gender_ratio": "Equal",
            "prevalence": "~30% globally; up to 80-90% in East Asian young adults",
            "geographic": "Highest in East Asia (Singapore, South Korea, China, Japan)"
        },
        "foods": [
            "Vitamin D-rich foods (fortified milk, eggs, fatty fish)",
            "Foods rich in dopamine precursors (bananas, almonds)",
            "Antioxidant-rich foods (berries, dark chocolate)",
            "Zinc-rich foods (pumpkin seeds, chickpeas)",
            "Vitamin A sources (sweet potatoes, carrots)",
            "Adequate protein for eye tissue health"
        ]
    },

    "Other Diseases": {
        "description": "This category covers other retinal pathologies including retinal dystrophies, macular holes, epiretinal membranes, central serous chorioretinopathy, and other less common conditions requiring specialist evaluation.",
        "severity": "Variable",
        "urgency": "Moderate",
        "color": "#607D8B",
        "symptoms": [
            "Varies by specific condition",
            "Visual disturbances or blurring",
            "Central or peripheral vision changes",
            "Floaters or flashes of light",
            "Color vision abnormalities",
            "Metamorphopsia (visual distortion)"
        ],
        "risk_factors": [
            "Varies by specific condition",
            "Age-related changes",
            "Genetic predisposition",
            "Prior eye surgery or trauma",
            "Systemic inflammatory conditions",
            "Medication side effects"
        ],
        "treatment": [
            "Treatment depends on exact diagnosis",
            "Comprehensive ophthalmic evaluation required",
            "May include observation, medication, or surgery",
            "OCT imaging for detailed structural assessment",
            "Fluorescein angiography for vascular assessment",
            "Referral to retina specialist recommended"
        ],
        "precautions": [
            "Seek comprehensive evaluation by a retina specialist",
            "Document all visual symptoms with dates and descriptions",
            "Bring all current medication lists to appointments",
            "Follow up as recommended by your ophthalmologist",
            "Report any new or worsening symptoms promptly",
            "Maintain a healthy lifestyle to support eye health",
            "Protect eyes from injury and UV exposure",
            "Keep records of all eye examinations and test results"
        ],
        "demographics": {
            "common_age": "Varies widely by condition",
            "gender_ratio": "Varies by condition",
            "prevalence": "Varies by specific disease",
            "geographic": "Worldwide distribution"
        },
        "foods": [
            "Well-balanced diet with plenty of fruits and vegetables",
            "Omega-3 fatty acids (fish, walnuts, flaxseeds)",
            "Antioxidant-rich foods (colorful fruits and vegetables)",
            "Adequate hydration (8 glasses of water daily)",
            "Lutein and zeaxanthin sources (eggs, corn, kale)",
            "Limit processed foods and excessive sugar"
        ]
    }
}


# ════════════════════════════════════════════════════════════════════
# AGENT 1: REPORT GENERATOR
# ════════════════════════════════════════════════════════════════════

class ReportAgent:
    """Generates comprehensive medical analysis reports."""

    def generate(self, result, quality, patient_id=None):
        disease = result["predicted_class"]
        confidence = result["confidence"]
        info = DISEASE_KNOWLEDGE_BASE.get(disease, {})
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pid = patient_id if patient_id else "ANONYMOUS"

        # Build model table
        model_lines = []
        for name, res in result.get("individual", {}).items():
            model_lines.append(
                f"| {name.title():20s} | {res['prediction']:30s} | {res['confidence']:.1%}       |"
            )
        model_table = chr(10).join(model_lines) if model_lines else "| No individual data available |  |  |"

        consensus = "ALL 3 MODELS AGREE" if result.get("unanimous", False) else "PARTIAL AGREEMENT"

        # Top 3 differential diagnoses
        probs = result.get("probabilities", [])
        top3_lines = []
        class_names = [
            "Normal", "Diabetic Retinopathy", "ARMD", "Media Haze",
            "Optic Disc Disease", "Retinal Vascular Disease", "Myopia", "Other Diseases"
        ]
        if len(probs) > 0:
            sorted_idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
            for rank, idx in enumerate(sorted_idx, 1):
                top3_lines.append(
                    f"| {rank}    | {class_names[idx]:30s} | {probs[idx]:.1%}       |"
                )
        top3_table = chr(10).join(top3_lines) if top3_lines else "| No probability data |  |  |"

        severity = info.get("severity", "N/A")
        urgency = info.get("urgency", "N/A")
        description = info.get("description", "No description available.")
        q_score = quality.get("score", "N/A")
        q_res = quality.get("resolution", "N/A")
        q_issues = ", ".join(quality.get("issues", ["None"]))
        report_id = random.randint(10000, 99999)

        if disease == "Normal":
            action1 = "No immediate intervention required. Continue routine screening."
            action2 = "Maintain annual eye examinations."
            action3 = "Continue healthy lifestyle practices."
        else:
            action1 = "Schedule follow-up with ophthalmologist within 1-2 weeks."
            action2 = "Consider additional diagnostic imaging (OCT, FA)."
            action3 = "Begin or optimize treatment as per clinical guidelines."

        report = f"""
# MediAgent-Retina Medical Analysis Report

---

| Field              | Value                                      |
|--------------------|---------------------------------------------|
| **Report ID**      | RPT-{report_id}                             |
| **Patient ID**     | {pid}                                       |
| **Date & Time**    | {timestamp}                                 |
| **System**         | MediAgent-Retina v2.0 (3-Model Ensemble)    |

---

## Primary Diagnosis

| Field              | Value                                      |
|--------------------|---------------------------------------------|
| **Condition**      | **{disease}**                               |
| **Confidence**     | **{confidence:.1%}**                        |
| **Severity**       | {severity}                                  |
| **Clinical Urgency** | {urgency}                                 |

### Description
{description}

---

## Individual Model Predictions

| Model                | Prediction                          | Confidence |
|----------------------|-------------------------------------|------------|
{model_table}

**Model Consensus:** {consensus}

---

## Top 3 Differential Diagnoses

| Rank | Condition                          | Probability |
|------|------------------------------------|-------------|
{top3_table}

---

## Image Quality Assessment

| Metric            | Value                              |
|-------------------|------------------------------------|
| **Quality Score** | {q_score}/100                      |
| **Resolution**    | {q_res}                            |
| **Issues**        | {q_issues}                         |

---

## Recommended Actions

1. {action1}
2. {action2}
3. {action3}

---

> **DISCLAIMER:** This AI-generated report is for screening purposes only.
> It does NOT constitute a medical diagnosis. Always consult a qualified
> ophthalmologist for clinical decisions.

---
*Generated by MediAgent-Retina v2.0 | {timestamp}*
"""
        return report


# ════════════════════════════════════════════════════════════════════
# AGENT 2: PRECAUTIONS & SAFETY ADVISOR
# ════════════════════════════════════════════════════════════════════

class PrecautionsAgent:
    """Generates disease-specific precautions and safety advice."""

    def generate(self, disease):
        info = DISEASE_KNOWLEDGE_BASE.get(disease, {})
        precautions = info.get("precautions", ["Consult your ophthalmologist."])
        risk_factors = info.get("risk_factors", ["Unknown"])
        foods = info.get("foods", ["Balanced diet recommended"])

        precautions_list = chr(10).join([f"- {p}" for p in precautions])
        risk_list = chr(10).join([f"- {r}" for r in risk_factors])
        food_list = chr(10).join([f"- {f}" for f in foods])

        urgency = info.get("urgency", "Moderate")
        if urgency == "URGENT":
            urgency_msg = "**URGENT** - Seek immediate ophthalmologic consultation!"
        elif urgency == "HIGH":
            urgency_msg = "**HIGH PRIORITY** - Schedule appointment within 1-2 weeks."
        elif urgency == "Moderate":
            urgency_msg = "**MODERATE** - Follow up within 1-3 months."
        else:
            urgency_msg = "**ROUTINE** - Annual screening recommended."

        output = f"""
## Precautions and Safety Guidelines for {disease}

### Urgency Level
{urgency_msg}

---

### Recommended Precautions
{precautions_list}

---

### Risk Factors to Monitor
{risk_list}

---

### Dietary Recommendations
{food_list}

---

### Lifestyle Recommendations
- **Exercise:** 30 minutes of moderate activity, 5 days per week
- **Sleep:** 7-8 hours of quality sleep per night
- **Smoking:** Quit immediately - smoking damages retinal blood vessels
- **Alcohol:** Limit to moderate consumption
- **Screen Time:** Follow the 20-20-20 rule
- **Sun Protection:** Wear UV-protective sunglasses outdoors

---

> **Remember:** Early detection and consistent follow-up are the keys
> to preserving your vision. Never ignore changes in your eyesight.
"""
        return output


# ════════════════════════════════════════════════════════════════════
# AGENT 3: DISEASE DETAILS
# ════════════════════════════════════════════════════════════════════

class DiseaseDetailsAgent:
    """Provides comprehensive disease information."""

    def generate(self, disease):
        info = DISEASE_KNOWLEDGE_BASE.get(disease, {})
        symptoms = info.get("symptoms", ["Information not available"])
        treatments = info.get("treatment", ["Consult specialist"])
        risk_factors = info.get("risk_factors", ["Unknown"])

        symptom_list = chr(10).join([f"- {s}" for s in symptoms])
        treatment_list = chr(10).join([f"- {t}" for t in treatments])
        risk_list = chr(10).join([f"- {r}" for r in risk_factors])

        severity = info.get("severity", "N/A")
        urgency = info.get("urgency", "N/A")
        description = info.get("description", "No description available.")

        if disease != "Normal":
            prognosis = (
                "With early detection and proper management, most patients can maintain "
                "functional vision. Regular monitoring is essential to track disease "
                "progression and treatment response."
            )
            emergency_intro = "Seek **immediate** medical attention if you experience:"
            emergency_extra = "Any sudden change related to your diagnosed condition"
        else:
            prognosis = (
                "Healthy retina detected. Continue preventive care. Annual comprehensive "
                "eye exams are recommended to maintain eye health."
            )
            emergency_intro = "Visit an eye doctor if you notice:"
            emergency_extra = "Any unexplained visual disturbance"

        output = f"""
## Comprehensive Information: {disease}

### Overview
{description}

| Property       | Value           |
|----------------|-----------------|
| **Severity**   | {severity}      |
| **Urgency**    | {urgency}       |

---

### Common Symptoms
{symptom_list}

---

### Risk Factors
{risk_list}

---

### Treatment Options
{treatment_list}

---

### Prognosis
{prognosis}

---

### When to Seek Emergency Care
{emergency_intro}
- Sudden vision loss or significant vision change
- New onset of flashes of light or shower of floaters
- Curtain-like shadow over your visual field
- Severe eye pain or redness
- {emergency_extra}

---

> **Note:** This information is for educational purposes. Individual cases
> may vary significantly. Always rely on your ophthalmologist for personalized advice.
"""
        return output


# ════════════════════════════════════════════════════════════════════
# AGENT 4: DEMOGRAPHICS & AGE GROUPS
# ════════════════════════════════════════════════════════════════════

class DemographicsAgent:
    """Provides demographic and epidemiological information."""

    def generate(self, disease):
        info = DISEASE_KNOWLEDGE_BASE.get(disease, {})
        demo = info.get("demographics", {})

        common_age = demo.get("common_age", "Variable")
        gender_ratio = demo.get("gender_ratio", "Equal")
        prevalence = demo.get("prevalence", "Data varies")
        geographic = demo.get("geographic", "Worldwide")

        if disease != "Normal":
            impact = (
                "This condition affects millions worldwide and is a significant "
                "cause of preventable blindness."
            )
            trend = (
                f"The prevalence of {disease} has been increasing globally due to "
                "aging populations, lifestyle changes, and improved diagnostic capabilities."
            )
        else:
            impact = (
                "Regular eye screening helps maintain ocular health across all demographics."
            )
            trend = (
                "Maintaining a normal retinal status requires ongoing preventive care "
                "throughout life."
            )

        # Disease-specific high-risk populations
        high_risk_map = {
            "Diabetic Retinopathy": """
- People with Type 1 or Type 2 diabetes for more than 10 years
- Patients with poorly controlled HbA1c (above 8%)
- Pregnant women with pre-existing diabetes
- People of South Asian, African, or Hispanic descent
- Patients with concurrent hypertension or kidney disease
""",
            "ARMD": """
- Adults over 60 years of age
- People with a family history of AMD
- Current or former smokers
- People of European descent
- Individuals with light-colored irises
- Those with cardiovascular disease
""",
            "Optic Disc Disease": """
- People of African descent (glaucoma risk 4-5x higher)
- Adults over 40 with family history of glaucoma
- Patients with elevated intraocular pressure
- Highly myopic individuals
- People with thin corneas
""",
            "Retinal Vascular Disease": """
- People with uncontrolled hypertension
- Diabetic patients
- Adults over 50 with cardiovascular disease
- Smokers
- People with blood clotting disorders
""",
            "Myopia": """
- Children of myopic parents (genetic risk)
- East Asian populations (80-90% prevalence in young adults)
- Children with excessive near work and limited outdoor time
- University students and professionals with intensive reading
""",
        }

        high_risk = high_risk_map.get(disease, """
- People over 40 years of age
- Those with family history of eye diseases
- Individuals with systemic health conditions
- Anyone experiencing changes in vision
""")

        if disease != "Normal":
            risk_intro = "Based on epidemiological data, the following groups should be particularly vigilant:"
        else:
            risk_intro = "The following groups benefit most from regular screening:"

        output = f"""
## Demographics and Epidemiology: {disease}

### Key Statistics

| Demographic Factor     | Details                    |
|------------------------|----------------------------|
| **Common Age Group**   | {common_age}               |
| **Gender Distribution**| {gender_ratio}             |
| **Global Prevalence**  | {prevalence}               |
| **Geographic Pattern** | {geographic}               |

---

### Global Impact
{impact}

### Trends
{trend}

---

### High-Risk Populations
{risk_intro}
{high_risk}

---

### Screening Recommendations

| Age Group    | Recommendation                              |
|--------------|----------------------------------------------|
| 20-39 years  | Complete eye exam every 5-10 years            |
| 40-54 years  | Complete eye exam every 2-4 years             |
| 55-64 years  | Complete eye exam every 1-3 years             |
| 65+ years    | Complete eye exam every 1-2 years             |
| Diabetic     | Annual dilated eye exam (minimum)             |
| High Risk    | As recommended by ophthalmologist             |

---

> Data based on global epidemiological studies. Individual risk may vary.
"""
        return output


# ════════════════════════════════════════════════════════════════════
# AGENT 5: INTERACTIVE CHATBOT
# ════════════════════════════════════════════════════════════════════


class ChatbotAgent:
    """Interactive chatbot for patient questions about eye health."""

    def __init__(self):
        self.greetings = [
            "Hello! I am MediBot, your eye health assistant. ",
            "Hi there! I am here to help with your eye health questions. ",
            "Welcome! I am MediBot, ready to assist you. ",
        ]

    def get_response(self, question, current_disease="Normal"):
        q = question.lower().strip()
        info = DISEASE_KNOWLEDGE_BASE.get(current_disease, {})

        # Greeting
        if any(word in q for word in ["hello", "hi", "hey", "greet"]):
            greeting = random.choice(self.greetings)
            return (
                f"{greeting}"
                f"I see the analysis detected **{current_disease}**. "
                "How can I help you understand this better?"
            )

        # Symptoms
        if any(word in q for word in ["symptom", "sign", "feel", "notice"]):
            symptoms = info.get("symptoms", ["No specific symptoms listed."])
            lines = chr(10).join([f"- {s}" for s in symptoms])
            return (
                f"### Common Symptoms of {current_disease}\n\n"
                f"{lines}\n\n"
                "**Important:** If you experience any sudden changes in vision, "
                "seek immediate medical attention."
            )

        # Treatment
        if any(word in q for word in ["treat", "cure", "therap", "medicat",
                                       "medicine", "drug", "surgery"]):
            treatments = info.get("treatment", ["Consult your ophthalmologist."])
            lines = chr(10).join([f"- {t}" for t in treatments])
            return (
                f"### Treatment Options for {current_disease}\n\n"
                f"{lines}\n\n"
                "**Note:** Treatment plans should be personalized by your "
                "ophthalmologist based on your specific condition."
            )

        # Prevention / Precautions
        if any(word in q for word in ["prevent", "avoid", "precaution", "protect",
                                       "safe", "care"]):
            precautions = info.get("precautions", ["Maintain regular eye checkups."])
            lines = chr(10).join([f"- {p}" for p in precautions])
            return (
                f"### Prevention and Precautions for {current_disease}\n\n"
                f"{lines}\n\n"
                "Prevention is always better than cure!"
            )

        # Food / Diet
        if any(word in q for word in ["food", "diet", "eat", "nutrition",
                                       "vitamin", "supplement"]):
            foods = info.get("foods", ["Balanced diet recommended."])
            lines = chr(10).join([f"- {f}" for f in foods])
            return (
                f"### Dietary Recommendations for Eye Health\n\n"
                f"For **{current_disease}**, consider:\n\n"
                f"{lines}\n\n"
                "A healthy diet supports overall eye health and may slow "
                "disease progression."
            )

        # Risk factors
        if any(word in q for word in ["risk", "cause", "why", "reason", "factor"]):
            risks = info.get("risk_factors", ["Various factors may contribute."])
            lines = chr(10).join([f"- {r}" for r in risks])
            return (
                f"### Risk Factors for {current_disease}\n\n"
                f"{lines}\n\n"
                "Understanding your risk factors helps in prevention and "
                "early detection."
            )

        # Doctor / When to see
        if any(word in q for word in ["doctor", "specialist", "ophthalmol",
                                       "hospital", "clinic", "emergency",
                                       "when should", "visit"]):
            urgency = info.get("urgency", "Moderate")
            return (
                f"### When to See a Doctor\n\n"
                f"For **{current_disease}** (Urgency: **{urgency}**):\n\n"
                "- **Immediately** if you have sudden vision loss, flashes, "
                "or a curtain-like shadow\n"
                "- **Within 1-2 weeks** if you notice gradual vision changes\n"
                "- **Regular follow-up** as recommended by your ophthalmologist\n\n"
                "Do not delay seeking care. Early treatment preserves vision!"
            )

        # What is / explain
        if any(word in q for word in ["what is", "explain", "tell me about",
                                       "describe", "information", "about",
                                       "details"]):
            desc = info.get("description", "No detailed description available.")
            severity = info.get("severity", "N/A")
            urgency = info.get("urgency", "N/A")
            return (
                f"### About {current_disease}\n\n"
                f"{desc}\n\n"
                f"**Severity:** {severity}\n"
                f"**Urgency:** {urgency}\n\n"
                "Would you like to know about symptoms, treatment, or "
                "prevention? Just ask!"
            )

        # Prognosis / outlook
        if any(word in q for word in ["prognos", "outlook", "future", "progress",
                                       "will i", "can i", "recover", "blind"]):
            if current_disease == "Normal":
                return (
                    "### Prognosis\n\n"
                    "Great news! Your retinal scan appears **normal**. "
                    "With continued healthy habits and regular checkups, "
                    "you can maintain good eye health.\n\n"
                    "Continue annual eye exams for ongoing monitoring."
                )
            return (
                f"### Prognosis for {current_disease}\n\n"
                "The outlook depends on several factors:\n\n"
                "- **Stage at detection** - Earlier is better\n"
                "- **Treatment compliance** - Following your doctors plan\n"
                "- **Underlying health** - Managing diabetes, hypertension\n"
                "- **Lifestyle factors** - Diet, exercise, smoking cessation\n\n"
                "Many patients maintain functional vision with proper management. "
                "Your ophthalmologist can give you a personalized prognosis.\n\n"
                "Stay positive and proactive about your eye health!"
            )

        # Thank you
        if any(word in q for word in ["thank", "thanks", "helpful", "appreciate"]):
            return (
                "You are welcome! I am glad I could help. Remember:\n\n"
                "- Follow your doctors recommendations\n"
                "- Keep your scheduled appointments\n"
                "- Do not hesitate to ask if you have more questions\n\n"
                "Take care of your eyes!"
            )

        # Default fallback
        return (
            f"Thank you for your question about **{current_disease}**.\n\n"
            "I can help you with:\n\n"
            "- **Symptoms** - What signs to look for\n"
            "- **Treatment** - Available treatment options\n"
            "- **Prevention** - How to protect your eyes\n"
            "- **Diet** - Foods that support eye health\n"
            "- **Risk factors** - What increases your risk\n"
            "- **When to see a doctor** - Urgency guidance\n\n"
            "Try asking something like:\n"
            f"_What are the symptoms of {current_disease}?_ or "
            f"_How is {current_disease} treated?_"
        )
