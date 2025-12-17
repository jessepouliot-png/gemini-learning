#!/usr/bin/env python3
"""
Example: Using Arch Optimizer Modules Programmatically

This example shows how to use the arch_optimizer modules in your own Python code.
"""

import os
from arch_optimizer import ArchOptimizer, SystemInfoCollector

def example_system_info_only():
    """Example: Collect system information without AI analysis."""
    print("=" * 80)
    print("Example 1: System Information Collection Only")
    print("=" * 80)
    
    # Create a system info collector
    collector = SystemInfoCollector()
    
    # Collect all system information
    info = collector.collect_all()
    
    # Display formatted summary
    print(collector.get_formatted_summary())
    
    # Access specific information
    print("\nDetailed CPU Info:")
    print(f"  Model: {info['cpu'].get('model', 'Unknown')}")
    print(f"  Cores: {info['cpu'].get('count_logical', 'Unknown')}")
    print(f"  Usage: {info['cpu'].get('usage_average', 0):.1f}%")


def example_with_ai_analysis():
    """Example: Full analysis with Gemini AI (requires API key)."""
    print("\n" + "=" * 80)
    print("Example 2: Full AI-Powered Analysis")
    print("=" * 80)
    
    # Check if API key is available
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️  Skipping AI analysis - GEMINI_API_KEY not set")
        print("To run this example, set your API key:")
        print("  export GEMINI_API_KEY='your-api-key-here'")
        return
    
    # Create optimizer with specific configuration
    config = {
        'analysis_depth': 'basic',
        'focus_areas': ['performance', 'memory'],
        'output_format': 'markdown'
    }
    
    optimizer = ArchOptimizer(config=config)
    
    # Collect system information
    print("Collecting system information...")
    optimizer.collect_system_info()
    
    # Perform AI analysis
    print("Analyzing with Gemini AI...")
    result = optimizer.analyze(
        focus_areas=['performance', 'memory', 'disk'],
        analysis_depth='basic'
    )
    
    if result['success']:
        print("✓ Analysis successful!")
        
        # Print summary
        optimizer.print_summary()
        
        # Generate report
        report_file = optimizer.generate_report('example_report.md', format='markdown')
        print(f"\n✓ Report saved to: {report_file}")
    else:
        print(f"✗ Analysis failed: {result.get('error')}")


def example_custom_focus():
    """Example: Focused analysis on specific areas."""
    print("\n" + "=" * 80)
    print("Example 3: Custom Focus Area Analysis")
    print("=" * 80)
    
    if not os.getenv('GEMINI_API_KEY'):
        print("⚠️  Skipping - GEMINI_API_KEY not set")
        return
    
    optimizer = ArchOptimizer()
    
    # Collect info
    optimizer.collect_system_info()
    
    # Focus only on security and services
    result = optimizer.analyze(
        focus_areas=['security', 'services'],
        analysis_depth='standard'
    )
    
    if result['success']:
        # Access parsed recommendations
        if optimizer.parsed_recommendations:
            high_impact = optimizer.parsed_recommendations.get('high', [])
            print(f"\nFound {len(high_impact)} high-impact recommendations")
            
            # Generate JSON report for programmatic use
            optimizer.generate_report('security_audit.json', format='json')
            print("✓ JSON report saved for programmatic processing")


if __name__ == '__main__':
    # Example 1: System info only (no API key needed)
    example_system_info_only()
    
    # Example 2: Full analysis with AI
    example_with_ai_analysis()
    
    # Example 3: Custom focus areas
    example_custom_focus()
    
    print("\n" + "=" * 80)
    print("Examples completed!")
    print("=" * 80)
