"""
Example script showing how to generate datasets and queries.
Place this in the examples/ folder.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from qdrant_benchmark.data_generator import DatasetGenerator
from qdrant_benchmark.query_generator import QueryGenerator


def generate_balanced_dataset():
    """Generate a balanced dataset across all domains."""
    print("=" * 60)
    print("Generating Balanced Dataset (10,000 items)")
    print("=" * 60)
    
    generator = DatasetGenerator(seed=42)
    
    # Equal distribution across domains
    domain_mix = {
        'tech': 0.25,
        'medical': 0.25,
        'pharmaceutical': 0.25,
        'health_insurance': 0.25
    }
    
    dataset = generator.generate(n=10000, domain_mix=domain_mix)
    generator.save_dataset(dataset, 'data/balanced_dataset_10k.json')
    
    # Print sample
    print("\nSample items:")
    for i, item in enumerate(dataset[:3], 1):
        print(f"\n{i}. [{item['domain'].upper()}] {item['title']}")
        print(f"   {item['description'][:100]}...")


def generate_tech_focused_dataset():
    """Generate a tech-focused dataset."""
    print("\n" + "=" * 60)
    print("Generating Tech-Focused Dataset (10,000 items)")
    print("=" * 60)
    
    generator = DatasetGenerator(seed=42)
    
    # 70% tech, 30% other
    domain_mix = {
        'tech': 0.70,
        'medical': 0.10,
        'pharmaceutical': 0.10,
        'health_insurance': 0.10
    }
    
    dataset = generator.generate(n=10000, domain_mix=domain_mix)
    generator.save_dataset(dataset, 'data/tech_focused_dataset_10k.json')


def generate_large_dataset():
    """Generate a large 100k dataset."""
    print("\n" + "=" * 60)
    print("Generating Large Dataset (100,000 items)")
    print("=" * 60)
    
    generator = DatasetGenerator(seed=42)
    
    domain_mix = {
        'tech': 0.25,
        'medical': 0.25,
        'pharmaceutical': 0.25,
        'health_insurance': 0.25
    }
    
    dataset = generator.generate(n=100000, domain_mix=domain_mix)
    generator.save_dataset(dataset, 'data/large_dataset_100k.json')


def generate_test_queries():
    """Generate test queries with both auto and manual."""
    print("\n" + "=" * 60)
    print("Generating Test Queries")
    print("=" * 60)
    
    generator = QueryGenerator(seed=42)
    
    # Generate 20 auto queries
    domain_mix = {
        'tech': 0.25,
        'medical': 0.25,
        'pharmaceutical': 0.25,
        'health_insurance': 0.25
    }
    
    auto_queries = generator.generate_auto_queries(n=20, domain_mix=domain_mix)
    generator.add_manual_queries(auto_queries)
    
    # Add some specific manual queries
    manual_queries = [
        "python machine learning best practices",
        "comprehensive cardiology treatment protocols",
        "antibiotic resistance management strategies",
        "affordable family health insurance with dental coverage",
        "javascript web development security patterns"
    ]
    
    generator.add_manual_queries(manual_queries)
    
    # Save and display
    metadata = {
        'auto_generated': 20,
        'manual_added': len(manual_queries),
        'domain_mix': domain_mix
    }
    
    generator.save_queries('data/test_queries.json', metadata=metadata)
    generator.display_queries(max_display=15)
    
    # Show distribution
    distribution = generator.get_domain_distribution()
    print(f"\n{'='*60}")
    print("Query Distribution:")
    print(f"{'='*60}")
    for domain, count in sorted(distribution.items()):
        if count > 0:
            print(f"  {domain}: {count}")


def load_and_use_data():
    """Example of loading previously generated data."""
    print("\n" + "=" * 60)
    print("Loading Previously Generated Data")
    print("=" * 60)
    
    # Load dataset
    dataset = DatasetGenerator.load_dataset('data/balanced_dataset_10k.json')
    print(f"Dataset has {len(dataset)} items")
    
    # Load queries
    query_gen = QueryGenerator()
    queries = query_gen.load_queries('data/test_queries.json')
    print(f"Loaded {len(queries)} test queries")
    
    return dataset, queries


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate datasets and queries')
    parser.add_argument('--all', action='store_true',
                       help='Generate all datasets and queries')
    parser.add_argument('--balanced', action='store_true',
                       help='Generate balanced 10k dataset')
    parser.add_argument('--tech', action='store_true',
                       help='Generate tech-focused dataset')
    parser.add_argument('--large', action='store_true',
                       help='Generate large 100k dataset')
    parser.add_argument('--queries', action='store_true',
                       help='Generate test queries')
    parser.add_argument('--load', action='store_true',
                       help='Load and display existing data')
    
    args = parser.parse_args()
    
    # If no specific flags, show help
    if not any([args.all, args.balanced, args.tech, args.large, args.queries, args.load]):
        parser.print_help()
        sys.exit(0)
    
    # Create data directory
    Path('data').mkdir(exist_ok=True)
    
    if args.all or args.balanced:
        generate_balanced_dataset()
    
    if args.all or args.tech:
        generate_tech_focused_dataset()
    
    if args.all or args.large:
        generate_large_dataset()
    
    if args.all or args.queries:
        generate_test_queries()
    
    if args.load:
        load_and_use_data()
    
    print("\n✓ Done!")