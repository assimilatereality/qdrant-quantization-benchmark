"""
Query generator for creating test queries across multiple domains.
Supports auto-generation and manual curation of test queries.
"""

import json
import random
from typing import List, Dict, Any
from pathlib import Path


class QueryGenerator:
    """Generate and manage test queries for benchmarking."""
    
    def __init__(self, seed: int = 42):
        """Initialize query generator with random seed."""
        random.seed(seed)
        self.queries = []
    
    def generate_auto_queries(self, n: int = 20, domain_mix: Dict[str, float] = None) -> List[str]:
        """
        Auto-generate queries distributed across domains.
        
        Args:
            n: Number of queries to generate
            domain_mix: Dictionary of domain weights
        
        Returns:
            List of query strings
        """
        if domain_mix is None:
            domain_mix = {
                'tech': 0.25,
                'medical': 0.25,
                'pharmaceutical': 0.25,
                'health_insurance': 0.25
            }
        
        # Normalize weights
        total_weight = sum(domain_mix.values())
        domain_mix = {k: v/total_weight for k, v in domain_mix.items()}
        
        queries = []
        
        # Calculate queries per domain
        for domain, weight in domain_mix.items():
            count = int(n * weight)
            if domain == 'tech':
                queries.extend(self._generate_tech_queries(count))
            elif domain == 'medical':
                queries.extend(self._generate_medical_queries(count))
            elif domain == 'pharmaceutical':
                queries.extend(self._generate_pharmaceutical_queries(count))
            elif domain == 'health_insurance':
                queries.extend(self._generate_insurance_queries(count))
        
        # Fill remaining slots
        while len(queries) < n:
            domain = random.choice(list(domain_mix.keys()))
            if domain == 'tech':
                queries.extend(self._generate_tech_queries(1))
            elif domain == 'medical':
                queries.extend(self._generate_medical_queries(1))
            elif domain == 'pharmaceutical':
                queries.extend(self._generate_pharmaceutical_queries(1))
            elif domain == 'health_insurance':
                queries.extend(self._generate_insurance_queries(1))
        
        return queries[:n]
    
    def _generate_tech_queries(self, n: int) -> List[str]:
        """Generate tech-related queries."""
        templates = [
            "{language} {topic} tutorial",
            "learn {language} programming {topic}",
            "{difficulty} {topic} and {topic2}",
            "{language} for {topic}",
            "{topic} best practices {language}",
            "advanced {topic} techniques",
            "{language} {topic} examples",
            "beginner guide to {language}",
            "{topic} patterns in {language}",
            "modern {language} development"
        ]
        
        languages = ["python", "javascript", "java", "rust", "go", "c++", "typescript"]
        topics = ["machine learning", "web development", "data science", "algorithms",
                 "security", "testing", "cloud computing", "API design", "microservices"]
        difficulties = ["beginner", "intermediate", "advanced"]
        
        queries = []
        for _ in range(n):
            template = random.choice(templates)
            query = template.format(
                language=random.choice(languages),
                topic=random.choice(topics),
                topic2=random.choice([t for t in topics if t != random.choice(topics)]),
                difficulty=random.choice(difficulties)
            )
            queries.append(query)
        
        return queries
    
    def _generate_medical_queries(self, n: int) -> List[str]:
        """Generate medical-related queries."""
        templates = [
            "{specialty} {topic} guide",
            "clinical {topic} for {specialty}",
            "{specialty} diagnosis and {topic}",
            "{topic} in {specialty} practice",
            "{specialty} patient {topic}",
            "evidence-based {specialty} {topic}",
            "{specialty} treatment protocols",
            "latest {specialty} research {topic}",
            "{topic} management in {specialty}",
            "{specialty} clinical guidelines"
        ]
        
        specialties = ["cardiology", "neurology", "oncology", "pediatrics", "surgery",
                      "radiology", "psychiatry", "orthopedics", "emergency medicine"]
        topics = ["treatment", "diagnosis", "management", "procedures", "guidelines",
                 "case studies", "pharmacotherapy", "interventions", "protocols"]
        
        queries = []
        for _ in range(n):
            template = random.choice(templates)
            query = template.format(
                specialty=random.choice(specialties),
                topic=random.choice(topics)
            )
            queries.append(query)
        
        return queries
    
    def _generate_pharmaceutical_queries(self, n: int) -> List[str]:
        """Generate pharmaceutical-related queries."""
        templates = [
            "{drug_class} for {condition}",
            "{condition} medication {form}",
            "{drug_class} side effects and dosing",
            "treatment for {condition}",
            "{form} medications for {condition}",
            "{drug_class} mechanism of action",
            "{condition} drug therapy",
            "prescription {drug_class}",
            "{drug_class} pharmacology",
            "{condition} pharmaceutical treatment"
        ]
        
        drug_classes = ["antibiotic", "antihypertensive", "analgesic", "antidepressant",
                       "anticoagulant", "bronchodilator", "antihistamine", "statin"]
        conditions = ["hypertension", "diabetes", "depression", "pain", "infection",
                     "asthma", "allergies", "high cholesterol", "anxiety"]
        forms = ["tablet", "capsule", "injection", "inhaler", "syrup"]
        
        queries = []
        for _ in range(n):
            template = random.choice(templates)
            query = template.format(
                drug_class=random.choice(drug_classes),
                condition=random.choice(conditions),
                form=random.choice(forms)
            )
            queries.append(query)
        
        return queries
    
    def _generate_insurance_queries(self, n: int) -> List[str]:
        """Generate health insurance-related queries."""
        templates = [
            "{tier} {plan_type} health insurance",
            "{coverage} health plan with {feature}",
            "affordable {plan_type} insurance",
            "{tier} tier health coverage",
            "{coverage} insurance with low deductible",
            "{plan_type} plans with {feature}",
            "comprehensive {coverage} coverage",
            "{tier} health insurance options",
            "{plan_type} with {feature} benefits",
            "best {tier} {coverage} plans"
        ]
        
        tiers = ["bronze", "silver", "gold", "platinum"]
        plan_types = ["HMO", "PPO", "EPO", "HDHP"]
        coverage_types = ["individual", "family", "employer", "student"]
        features = ["prescription coverage", "dental", "vision", "mental health",
                   "telehealth", "preventive care", "maternity"]
        
        queries = []
        for _ in range(n):
            template = random.choice(templates)
            query = template.format(
                tier=random.choice(tiers),
                plan_type=random.choice(plan_types),
                coverage=random.choice(coverage_types),
                feature=random.choice(features)
            )
            queries.append(query)
        
        return queries
    
    def add_manual_queries(self, queries: List[str]):
        """Add manually created queries to the collection."""
        self.queries.extend(queries)
        print(f"✓ Added {len(queries)} manual queries")
    
    def add_manual_query(self, query: str):
        """Add a single manual query."""
        self.queries.append(query)
        print(f"✓ Added query: '{query}'")
    
    def remove_query(self, query: str):
        """Remove a specific query."""
        if query in self.queries:
            self.queries.remove(query)
            print(f"✓ Removed query: '{query}'")
        else:
            print(f"✗ Query not found: '{query}'")
    
    def get_queries(self) -> List[str]:
        """Get all queries."""
        return self.queries
    
    def clear_queries(self):
        """Clear all queries."""
        self.queries = []
        print("✓ Cleared all queries")
    
    def save_queries(self, filepath: str, metadata: Dict[str, Any] = None):
        """Save queries to JSON file with optional metadata."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "queries": self.queries,
            "metadata": metadata or {},
            "count": len(self.queries)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"✓ Saved {len(self.queries)} queries to {filepath}")
    
    def load_queries(self, filepath: str) -> List[str]:
        """Load queries from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            # Legacy format: just a list of queries
            self.queries = data
        elif isinstance(data, dict) and 'queries' in data:
            # New format: dict with queries and metadata
            self.queries = data['queries']
        else:
            raise ValueError("Invalid query file format")
        
        print(f"✓ Loaded {len(self.queries)} queries from {filepath}")
        return self.queries
    
    def display_queries(self, max_display: int = None):
        """Display all queries (or first N queries)."""
        display_count = min(len(self.queries), max_display) if max_display else len(self.queries)
        
        print(f"\n{'='*60}")
        print(f"Test Queries ({display_count}/{len(self.queries)} shown)")
        print(f"{'='*60}")
        
        for i, query in enumerate(self.queries[:display_count], 1):
            print(f"{i:3d}. {query}")
        
        if max_display and len(self.queries) > max_display:
            print(f"\n... and {len(self.queries) - max_display} more queries")
    
    def get_domain_distribution(self) -> Dict[str, int]:
        """Analyze domain distribution of queries (basic keyword matching)."""
        distribution = {
            'tech': 0,
            'medical': 0,
            'pharmaceutical': 0,
            'health_insurance': 0,
            'unknown': 0
        }
        
        tech_keywords = ['python', 'javascript', 'programming', 'code', 'algorithm', 'web', 'api']
        medical_keywords = ['cardiology', 'surgery', 'clinical', 'patient', 'diagnosis', 'treatment']
        pharma_keywords = ['medication', 'drug', 'dosage', 'prescription', 'antibiotic', 'pharmaceutical']
        insurance_keywords = ['insurance', 'coverage', 'plan', 'hmo', 'ppo', 'deductible', 'premium']
        
        for query in self.queries:
            query_lower = query.lower()
            categorized = False
            
            if any(keyword in query_lower for keyword in tech_keywords):
                distribution['tech'] += 1
                categorized = True
            if any(keyword in query_lower for keyword in medical_keywords):
                distribution['medical'] += 1
                categorized = True
            if any(keyword in query_lower for keyword in pharma_keywords):
                distribution['pharmaceutical'] += 1
                categorized = True
            if any(keyword in query_lower for keyword in insurance_keywords):
                distribution['health_insurance'] += 1
                categorized = True
            
            if not categorized:
                distribution['unknown'] += 1
        
        return distribution


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate test queries')
    parser.add_argument('-n', '--num-queries', type=int, default=20,
                       help='Number of auto-generated queries (default: 20)')
    parser.add_argument('-o', '--output', type=str, default='data/queries.json',
                       help='Output filepath (default: data/queries.json)')
    parser.add_argument('--tech', type=float, default=0.25,
                       help='Proportion of tech queries (default: 0.25)')
    parser.add_argument('--medical', type=float, default=0.25,
                       help='Proportion of medical queries (default: 0.25)')
    parser.add_argument('--pharma', type=float, default=0.25,
                       help='Proportion of pharmaceutical queries (default: 0.25)')
    parser.add_argument('--insurance', type=float, default=0.25,
                       help='Proportion of health insurance queries (default: 0.25)')
    parser.add_argument('--manual', nargs='+', type=str,
                       help='Additional manual queries to include')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--display', action='store_true',
                       help='Display generated queries')
    
    args = parser.parse_args()
    
    domain_mix = {
        'tech': args.tech,
        'medical': args.medical,
        'pharmaceutical': args.pharma,
        'health_insurance': args.insurance
    }
    
    generator = QueryGenerator(seed=args.seed)
    
    # Generate auto queries
    auto_queries = generator.generate_auto_queries(n=args.num_queries, domain_mix=domain_mix)
    generator.add_manual_queries(auto_queries)
    
    # Add manual queries if provided
    if args.manual:
        generator.add_manual_queries(args.manual)
    
    # Save queries
    metadata = {
        'auto_generated': args.num_queries,
        'manual_added': len(args.manual) if args.manual else 0,
        'domain_mix': domain_mix,
        'seed': args.seed
    }
    generator.save_queries(args.output, metadata=metadata)
    
    # Display if requested
    if args.display:
        generator.display_queries()
        
        distribution = generator.get_domain_distribution()
        print(f"\n{'='*60}")
        print("Query Distribution by Domain:")
        print(f"{'='*60}")
        for domain, count in sorted(distribution.items()):
            if count > 0:
                print(f"  {domain}: {count}")