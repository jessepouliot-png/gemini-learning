#!/usr/bin/env python3
"""
Arch Linux Optimizer - Main Application

A Python application that uses Google's Gemini AI to analyze and optimize
an Arch Linux system.
"""

import argparse
import sys
import os
import json
from pathlib import Path
from arch_optimizer import ArchOptimizer, load_config


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze and optimize Arch Linux system using Gemini AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with API key from environment
  python arch_optimizer.py
  
  # Use custom configuration file
  python arch_optimizer.py --config my_config.json
  
  # Focus on specific areas
  python arch_optimizer.py --focus performance memory disk
  
  # Generate markdown report
  python arch_optimizer.py --format markdown --output report.md
  
  # Deep analysis
  python arch_optimizer.py --depth deep
  
Environment Variables:
  GEMINI_API_KEY    Your Google Gemini API key (required if not in config)
        """
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (JSON format)'
    )
    
    parser.add_argument(
        '--api-key', '-k',
        type=str,
        help='Gemini API key (overrides config and environment variable)'
    )
    
    parser.add_argument(
        '--focus', '-f',
        nargs='+',
        choices=['performance', 'memory', 'disk', 'services', 'packages', 'boot_time', 'security'],
        help='Specific areas to focus analysis on'
    )
    
    parser.add_argument(
        '--depth', '-d',
        type=str,
        choices=['basic', 'standard', 'deep'],
        default='standard',
        help='Analysis depth level (default: standard)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['text', 'markdown', 'json'],
        default='text',
        help='Output report format (default: text)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output report file path'
    )
    
    parser.add_argument(
        '--no-report',
        action='store_true',
        help='Skip generating report file (only print to console)'
    )
    
    parser.add_argument(
        '--system-info-only',
        action='store_true',
        help='Only collect and display system information (no AI analysis)'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='Arch Linux Optimizer 1.0.0'
    )
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = {}
    if args.config:
        try:
            config = load_config(args.config)
            print(f"Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"Error loading configuration: {e}", file=sys.stderr)
            return 1
    
    # Get API key (priority: argument > config > environment)
    api_key = args.api_key or config.get('api_key') or os.getenv('GEMINI_API_KEY')
    
    # If we're doing AI analysis, we need an API key
    if not args.system_info_only and not api_key:
        print("Error: Gemini API key is required for analysis.", file=sys.stderr)
        print("Provide it via:", file=sys.stderr)
        print("  1. --api-key argument", file=sys.stderr)
        print("  2. GEMINI_API_KEY environment variable", file=sys.stderr)
        print("  3. 'api_key' in configuration file", file=sys.stderr)
        print("\nOr use --system-info-only to just view system information.", file=sys.stderr)
        return 1
    
    try:
        # Initialize optimizer
        print("Initializing Arch Linux Optimizer...")
        # Only initialize Gemini client if not in system-info-only mode
        optimizer = ArchOptimizer(
            api_key=api_key, 
            config=config, 
            init_gemini=not args.system_info_only
        )
        
        # Collect system information
        print("\n" + "=" * 80)
        print("COLLECTING SYSTEM INFORMATION")
        print("=" * 80)
        optimizer.collect_system_info()
        
        # Display system summary
        print("\n" + optimizer.system_collector.get_formatted_summary())
        
        # If system-info-only mode, stop here
        if args.system_info_only:
            print("System information collection complete.")
            print("Skipping AI analysis (--system-info-only mode)")
            return 0
        
        # Perform analysis
        print("\n" + "=" * 80)
        print("PERFORMING AI ANALYSIS")
        print("=" * 80)
        
        # Get focus areas and depth
        focus_areas = args.focus or config.get('focus_areas')
        analysis_depth = args.depth or config.get('analysis_depth', 'standard')
        
        result = optimizer.analyze(
            focus_areas=focus_areas,
            analysis_depth=analysis_depth
        )
        
        if not result.get('success'):
            print(f"\nError during analysis: {result.get('error')}", file=sys.stderr)
            return 1
        
        # Print summary to console
        optimizer.print_summary()
        
        # Generate report unless --no-report is specified
        if not args.no_report:
            output_format = args.format or config.get('output_format', 'text')
            
            # Determine output file
            if args.output:
                output_file = args.output
            elif 'report_file' in config:
                output_file = config['report_file']
            else:
                # Default file based on format
                extensions = {'text': '.txt', 'markdown': '.md', 'json': '.json'}
                output_file = f'arch_optimization_report{extensions[output_format]}'
            
            print(f"\n" + "=" * 80)
            print("GENERATING REPORT")
            print("=" * 80)
            optimizer.generate_report(output_file=output_file, format=output_format)
            print(f"\nFull report saved to: {output_file}")
        
        print("\n" + "=" * 80)
        print("⚠️  IMPORTANT SAFETY REMINDER")
        print("=" * 80)
        print("1. Always backup your system before making changes")
        print("2. Review each recommendation carefully")
        print("3. Test in a safe environment when possible")
        print("4. Some recommendations may not apply to your use case")
        print("5. Keep records of changes for easy rollback")
        print("=" * 80)
        
        print("\n✓ Analysis complete!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        return 130
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
