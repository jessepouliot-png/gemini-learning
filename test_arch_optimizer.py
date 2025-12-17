#!/usr/bin/env python3
"""
Tests for Arch Linux Optimizer

Basic tests to validate functionality without requiring an API key.
"""

import sys
import json
from arch_optimizer import SystemInfoCollector, ArchOptimizer, load_config


def test_system_info_collector():
    """Test system information collection."""
    print("Testing SystemInfoCollector...")
    
    collector = SystemInfoCollector()
    info = collector.collect_all()
    
    # Check that all required sections are present
    required_sections = ['cpu', 'memory', 'disk', 'services', 'packages', 
                         'boot', 'network', 'kernel', 'timestamp']
    
    for section in required_sections:
        assert section in info, f"Missing section: {section}"
    
    # Check CPU info
    assert 'cpu' in info
    cpu = info['cpu']
    if 'error' not in cpu:
        assert 'count_logical' in cpu
        assert 'usage_average' in cpu
    
    # Check memory info
    assert 'memory' in info
    mem = info['memory']
    if 'error' not in mem:
        assert 'ram' in mem
        assert 'swap' in mem
    
    # Test formatted summary
    summary = collector.get_formatted_summary()
    assert len(summary) > 0
    assert 'System Information Summary' in summary
    
    print("✓ SystemInfoCollector tests passed")


def test_config_loading():
    """Test configuration loading."""
    print("Testing configuration loading...")
    
    # Test loading example config
    config = load_config('config.example.json')
    
    assert 'api_key' in config
    assert 'model' in config
    assert 'focus_areas' in config
    assert isinstance(config['focus_areas'], list)
    
    print("✓ Configuration loading tests passed")


def test_optimizer_without_api():
    """Test optimizer initialization without API key (system-info-only mode)."""
    print("Testing ArchOptimizer without API key...")
    
    # Initialize without Gemini client
    optimizer = ArchOptimizer(init_gemini=False)
    
    # Collect system info
    info = optimizer.collect_system_info()
    assert info is not None
    assert 'cpu' in info
    
    # Try to analyze without API key - should fail gracefully
    result = optimizer.analyze()
    assert not result['success']
    assert 'error' in result
    
    print("✓ ArchOptimizer tests passed")


def test_module_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    # These imports should work without errors
    from arch_optimizer import SystemInfoCollector
    from arch_optimizer import GeminiClient
    from arch_optimizer import ArchOptimizer
    from arch_optimizer import load_config
    
    # Check version
    import arch_optimizer
    assert hasattr(arch_optimizer, '__version__')
    
    print("✓ Module import tests passed")


def test_cli_help():
    """Test CLI help functionality."""
    print("Testing CLI help...")
    
    import subprocess
    result = subprocess.run(
        ['python', 'arch_optimizer.py', '--help'],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    assert result.returncode == 0
    assert 'Analyze and optimize Arch Linux' in result.stdout
    assert '--focus' in result.stdout
    assert '--system-info-only' in result.stdout
    
    print("✓ CLI help tests passed")


def test_cli_version():
    """Test CLI version flag."""
    print("Testing CLI version...")
    
    import subprocess
    result = subprocess.run(
        ['python', 'arch_optimizer.py', '--version'],
        capture_output=True,
        text=True,
        timeout=10
    )
    
    assert result.returncode == 0
    assert '1.0.0' in result.stdout
    
    print("✓ CLI version tests passed")


def test_requirements_coverage():
    """Test that all requirements are addressed."""
    print("Testing requirements coverage...")
    
    # Check that required files exist
    import os
    
    required_files = [
        'arch_optimizer.py',
        'arch_optimizer/__init__.py',
        'arch_optimizer/system_info.py',
        'arch_optimizer/gemini_client.py',
        'arch_optimizer/analyzer.py',
        'requirements.txt',
        'config.example.json',
        'README_ARCH_OPTIMIZER.md',
        '.env.example'
    ]
    
    for file in required_files:
        assert os.path.exists(file), f"Required file missing: {file}"
    
    # Check requirements.txt content
    with open('requirements.txt', 'r') as f:
        requirements = f.read()
        assert 'google-generativeai' in requirements
        assert 'psutil' in requirements
        assert 'python-dotenv' in requirements
    
    print("✓ Requirements coverage tests passed")


def run_all_tests():
    """Run all tests."""
    print("=" * 80)
    print("Running Arch Linux Optimizer Tests")
    print("=" * 80)
    print()
    
    tests = [
        test_module_imports,
        test_system_info_collector,
        test_config_loading,
        test_optimizer_without_api,
        test_cli_help,
        test_cli_version,
        test_requirements_coverage
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 80)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 80)
    
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
