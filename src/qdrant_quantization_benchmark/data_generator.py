"""
Data generator for creating diverse datasets across multiple domains.
Supports tech, medical, pharmaceutical, and health insurance items.
"""

import random
import json
from typing import List, Dict, Any
from pathlib import Path


class DatasetGenerator:
    """Generate synthetic data across multiple domains with minimal duplication."""
    
    def __init__(self, seed: int = 42):
        """Initialize generator with random seed for reproducibility."""
        random.seed(seed)
        self.domains = {
            'tech': self._generate_tech_item,
            'medical': self._generate_medical_item,
            'pharmaceutical': self._generate_pharmaceutical_item,
            'health_insurance': self._generate_health_insurance_item
        }
    
    def generate(self, n: int = 100, domain_mix: Dict[str, float] = None) -> List[Dict[str, Any]]:
        """
        Generate n items distributed across domains.
        
        Args:
            n: Total number of items to generate
            domain_mix: Dictionary of domain weights, e.g., {'tech': 0.4, 'medical': 0.6}
                       If None, distributes evenly across all domains
        
        Returns:
            List of dictionaries containing generated items
        """
        if domain_mix is None:
            # Even distribution across all domains
            domain_mix = {domain: 1.0 for domain in self.domains.keys()}
        
        # Normalize weights
        total_weight = sum(domain_mix.values())
        domain_mix = {k: v/total_weight for k, v in domain_mix.items()}
        
        # Calculate items per domain
        items_per_domain = {}
        remaining = n
        for domain, weight in domain_mix.items():
            count = int(n * weight)
            items_per_domain[domain] = count
            remaining -= count
        
        # Distribute remaining items
        if remaining > 0:
            for domain in list(items_per_domain.keys())[:remaining]:
                items_per_domain[domain] += 1
        
        # Generate items
        dataset = []
        item_id = 0
        
        for domain, count in items_per_domain.items():
            generator_func = self.domains.get(domain)
            if generator_func:
                for i in range(count):
                    item = generator_func(item_id, i)
                    dataset.append(item)
                    item_id += 1
        
        return dataset
    
    def _generate_tech_item(self, item_id: int, domain_index: int) -> Dict[str, Any]:
        """Generate tech/programming book item."""
        languages = ["Python", "JavaScript", "Java", "C++", "Ruby", "Go", "Rust", 
                    "TypeScript", "Kotlin", "Swift", "PHP", "C#", "Scala", "R"]
        
        topics = ["web development", "data science", "machine learning", "algorithms", 
                 "system design", "testing", "security", "DevOps", "mobile apps", 
                 "game development", "embedded systems", "cloud computing", "AI/ML",
                 "blockchain", "microservices", "containerization", "API design"]
        
        intro_phrases = [
            "Comprehensive guide to", "Learn", "Master", "Deep dive into",
            "Practical introduction to", "Advanced techniques for", "Getting started with",
            "Professional", "Essential", "Complete", "Hands-on", "Modern approach to"
        ]
        
        detail_phrases = [
            "Covers practical examples, best practices, and real-world applications.",
            "Includes hands-on projects and code samples throughout.",
            "Features expert insights and industry-tested patterns.",
            "Step-by-step tutorials with detailed explanations.",
            "Real-world case studies and production-ready code.",
            "Comprehensive coverage from basics to advanced concepts.",
            "Interactive exercises and project-based learning.",
            "Industry-standard techniques and cutting-edge practices.",
            "Battle-tested solutions for modern development challenges."
        ]
        
        audience_phrases = [
            "Suitable for intermediate developers",
            "Perfect for beginners and intermediate programmers",
            "Designed for experienced developers",
            "Ideal for software engineers",
            "Great for aspiring professionals",
            "Built for self-learners and bootcamp students",
            "Tailored for enterprise developers"
        ]
        
        lang = languages[domain_index % len(languages)]
        topic = topics[(domain_index // len(languages)) % len(topics)]
        
        intro = random.choice(intro_phrases)
        detail = random.choice(detail_phrases)
        audience = random.choice(audience_phrases)
        
        edition = (domain_index // (len(languages) * len(topics))) + 1
        year = 2020 + (domain_index % 5)
        
        return {
            "id": item_id,
            "domain": "tech",
            "title": f"{lang} for {topic.title()} - Edition {edition}",
            "description": f"{intro} {lang} programming with focus on {topic}. "
                         f"{detail} {audience} looking to master {topic}.",
            "metadata": {
                "language": lang,
                "topic": topic,
                "edition": edition,
                "pages": 200 + (domain_index * 7) % 300,
                "difficulty": ["beginner", "intermediate", "advanced"][domain_index % 3],
                "year": year
            }
        }
    
    def _generate_medical_item(self, item_id: int, domain_index: int) -> Dict[str, Any]:
        """Generate medical textbook/resource item."""
        specialties = ["Cardiology", "Neurology", "Oncology", "Pediatrics", "Surgery",
                      "Radiology", "Psychiatry", "Dermatology", "Orthopedics", "Endocrinology",
                      "Gastroenterology", "Pulmonology", "Nephrology", "Rheumatology"]
        
        topics = ["diagnosis", "treatment protocols", "patient management", "clinical practice",
                 "surgical techniques", "emergency medicine", "preventive care", "pathology",
                 "pharmacotherapy", "diagnostic imaging", "interventional procedures"]
        
        formats = ["Textbook", "Clinical Guide", "Reference Manual", "Handbook", "Atlas",
                  "Case Studies", "Review", "Protocols", "Guidelines"]
        
        audiences = ["medical students", "residents", "practicing physicians", "specialists",
                    "healthcare professionals", "clinical researchers"]
        
        specialty = specialties[domain_index % len(specialties)]
        topic = topics[(domain_index // len(specialties)) % len(topics)]
        format_type = formats[domain_index % len(formats)]
        audience = audiences[domain_index % len(audiences)]
        
        edition = (domain_index // (len(specialties) * 2)) + 1
        year = 2018 + (domain_index % 7)
        
        title = f"{specialty} {format_type}: {topic.title()} - {edition}th Edition"
        
        description = (f"Comprehensive {specialty.lower()} resource covering {topic}. "
                      f"Evidence-based approach with latest research and clinical guidelines. "
                      f"Designed for {audience} with practical applications and case examples. "
                      f"Updated with current standards of care and treatment recommendations.")
        
        return {
            "id": item_id,
            "domain": "medical",
            "title": title,
            "description": description,
            "metadata": {
                "specialty": specialty,
                "topic": topic,
                "format": format_type,
                "edition": edition,
                "pages": 400 + (domain_index * 11) % 600,
                "audience": audience,
                "year": year,
                "peer_reviewed": domain_index % 2 == 0
            }
        }
    
    def _generate_pharmaceutical_item(self, item_id: int, domain_index: int) -> Dict[str, Any]:
        """Generate pharmaceutical drug/medication item."""
        drug_classes = ["Antibiotic", "Antihypertensive", "Analgesic", "Antidepressant",
                       "Anticoagulant", "Bronchodilator", "Antihistamine", "Antidiarrheal",
                       "Immunosuppressant", "Anticonvulsant", "Antiviral", "Statin"]
        
        conditions = ["bacterial infections", "hypertension", "chronic pain", "depression",
                     "blood clots", "asthma", "allergies", "type 2 diabetes",
                     "autoimmune disorders", "epilepsy", "viral infections", "high cholesterol"]
        
        forms = ["tablet", "capsule", "injection", "syrup", "inhaler", "topical cream",
                "extended-release", "sublingual", "transdermal patch"]
        
        drug_class = drug_classes[domain_index % len(drug_classes)]
        condition = conditions[domain_index % len(conditions)]
        form = forms[domain_index % len(forms)]
        
        # Generate semi-realistic drug name
        prefixes = ["Ama", "Ben", "Car", "Dex", "Epo", "Flu", "Gab", "Hyd", "Ibu", "Ket"]
        suffixes = ["pine", "zole", "cin", "pril", "statin", "mab", "tinib", "oxin", "phen"]
        
        drug_name = (prefixes[domain_index % len(prefixes)] + 
                    suffixes[(domain_index // len(prefixes)) % len(suffixes)])
        
        dosages = ["10mg", "25mg", "50mg", "100mg", "200mg", "500mg", "1g"]
        dosage = dosages[domain_index % len(dosages)]
        
        title = f"{drug_name} {dosage} - {drug_class} ({form})"
        
        description = (f"{drug_class} medication used for treatment of {condition}. "
                      f"Available in {form} form. Mechanism of action targets specific pathways. "
                      f"Prescribed for adult patients under medical supervision. "
                      f"Standard dosing with documented efficacy and safety profile.")
        
        return {
            "id": item_id,
            "domain": "pharmaceutical",
            "title": title,
            "description": description,
            "metadata": {
                "drug_class": drug_class,
                "condition": condition,
                "dosage": dosage,
                "form": form,
                "generic_name": drug_name.lower(),
                "prescription_required": domain_index % 3 != 0,
                "controlled_substance": domain_index % 5 == 0
            }
        }
    
    def _generate_health_insurance_item(self, item_id: int, domain_index: int) -> Dict[str, Any]:
        """Generate health insurance plan item."""
        plan_types = ["HMO", "PPO", "EPO", "POS", "HDHP", "Catastrophic"]
        
        tiers = ["Bronze", "Silver", "Gold", "Platinum"]
        
        coverage_areas = ["Individual", "Family", "Medicare Supplement", "Short-term",
                         "Employer Group", "Student"]
        
        features = [
            "preventive care coverage", "prescription drug coverage", "mental health services",
            "dental and vision options", "telehealth services", "wellness programs",
            "specialist access", "emergency services", "hospitalization coverage",
            "maternity care", "rehabilitation services", "chronic disease management"
        ]
        
        plan_type = plan_types[domain_index % len(plan_types)]
        tier = tiers[domain_index % len(tiers)]
        coverage = coverage_areas[domain_index % len(coverage_areas)]
        
        # Select 3-5 features
        num_features = 3 + (domain_index % 3)
        selected_features = random.sample(features, min(num_features, len(features)))
        
        deductibles = [1000, 2000, 3000, 4000, 5000, 6000, 7500]
        deductible = deductibles[domain_index % len(deductibles)]
        
        title = f"{tier} {plan_type} - {coverage} Health Insurance Plan"
        
        features_text = ", ".join(selected_features)
        description = (f"{tier} tier {plan_type} health insurance plan for {coverage.lower()} coverage. "
                      f"Includes {features_text}. "
                      f"${deductible} annual deductible with comprehensive network access. "
                      f"Designed to balance affordability with quality healthcare access.")
        
        return {
            "id": item_id,
            "domain": "health_insurance",
            "title": title,
            "description": description,
            "metadata": {
                "plan_type": plan_type,
                "tier": tier,
                "coverage_type": coverage,
                "deductible": deductible,
                "features": selected_features,
                "network_size": ["Small", "Medium", "Large", "National"][domain_index % 4]
            }
        }
    
    def save_dataset(self, dataset: List[Dict[str, Any]], filepath: str):
        """Save dataset to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2)
        print(f"✓ Saved {len(dataset)} items to {filepath}")
    
    @staticmethod
    def load_dataset(filepath: str) -> List[Dict[str, Any]]:
        """Load dataset from JSON file."""
        with open(filepath, 'r') as f:
            dataset = json.load(f)
        print(f"✓ Loaded {len(dataset)} items from {filepath}")
        return dataset


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic datasets')
    parser.add_argument('-n', '--num-items', type=int, default=10000,
                       help='Number of items to generate (default: 10000)')
    parser.add_argument('-o', '--output', type=str, default='data/dataset.json',
                       help='Output filepath (default: data/dataset.json)')
    parser.add_argument('--tech', type=float, default=0.25,
                       help='Proportion of tech items (default: 0.25)')
    parser.add_argument('--medical', type=float, default=0.25,
                       help='Proportion of medical items (default: 0.25)')
    parser.add_argument('--pharma', type=float, default=0.25,
                       help='Proportion of pharmaceutical items (default: 0.25)')
    parser.add_argument('--insurance', type=float, default=0.25,
                       help='Proportion of health insurance items (default: 0.25)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    
    args = parser.parse_args()
    
    domain_mix = {
        'tech': args.tech,
        'medical': args.medical,
        'pharmaceutical': args.pharma,
        'health_insurance': args.insurance
    }
    
    generator = DatasetGenerator(seed=args.seed)
    dataset = generator.generate(n=args.num_items, domain_mix=domain_mix)
    generator.save_dataset(dataset, args.output)
    
    # Print statistics
    domain_counts = {}
    for item in dataset:
        domain = item['domain']
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    print("\nDataset Statistics:")
    print(f"  Total items: {len(dataset)}")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count} ({count/len(dataset)*100:.1f}%)")