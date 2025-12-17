#!/usr/bin/env python3
"""
System Information Collection Module for Arch Linux

This module collects comprehensive system information from an Arch Linux system
including CPU, memory, disk, services, packages, and more.
"""

import subprocess
import psutil
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional


class SystemInfoCollector:
    """Collects system information from Arch Linux."""

    def __init__(self):
        """Initialize the system information collector."""
        self.info = {}

    def collect_all(self) -> Dict[str, Any]:
        """
        Collect all system information.
        
        Returns:
            Dictionary containing all collected system information
        """
        self.info = {
            'timestamp': datetime.now().isoformat(),
            'cpu': self.get_cpu_info(),
            'memory': self.get_memory_info(),
            'disk': self.get_disk_info(),
            'services': self.get_systemd_services(),
            'packages': self.get_package_info(),
            'boot': self.get_boot_info(),
            'network': self.get_network_info(),
            'kernel': self.get_kernel_info()
        }
        return self.info

    def get_cpu_info(self) -> Dict[str, Any]:
        """
        Collect CPU usage and statistics.
        
        Returns:
            Dictionary containing CPU information
        """
        try:
            cpu_info = {
                'usage_percent': psutil.cpu_percent(interval=1, percpu=True),
                'usage_average': psutil.cpu_percent(interval=1),
                'count_physical': psutil.cpu_count(logical=False),
                'count_logical': psutil.cpu_count(logical=True),
                'frequency': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
                'load_average': os.getloadavg(),
            }
            
            # Get CPU model name from /proc/cpuinfo
            try:
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            cpu_info['model'] = line.split(':')[1].strip()
                            break
            except Exception:
                cpu_info['model'] = 'Unknown'
                
            return cpu_info
        except Exception as e:
            return {'error': str(e)}

    def get_memory_info(self) -> Dict[str, Any]:
        """
        Collect memory (RAM and swap) usage.
        
        Returns:
            Dictionary containing memory information
        """
        try:
            mem = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return {
                'ram': {
                    'total': mem.total,
                    'available': mem.available,
                    'used': mem.used,
                    'percent': mem.percent,
                    'total_gb': round(mem.total / (1024**3), 2),
                    'used_gb': round(mem.used / (1024**3), 2),
                    'available_gb': round(mem.available / (1024**3), 2)
                },
                'swap': {
                    'total': swap.total,
                    'used': swap.used,
                    'free': swap.free,
                    'percent': swap.percent,
                    'total_gb': round(swap.total / (1024**3), 2),
                    'used_gb': round(swap.used / (1024**3), 2)
                }
            }
        except Exception as e:
            return {'error': str(e)}

    def get_disk_info(self) -> Dict[str, Any]:
        """
        Collect disk usage and I/O statistics.
        
        Returns:
            Dictionary containing disk information
        """
        try:
            disk_info = {
                'partitions': [],
                'io_stats': {}
            }
            
            # Get partition information
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_info['partitions'].append({
                        'device': partition.device,
                        'mountpoint': partition.mountpoint,
                        'fstype': partition.fstype,
                        'total_gb': round(usage.total / (1024**3), 2),
                        'used_gb': round(usage.used / (1024**3), 2),
                        'free_gb': round(usage.free / (1024**3), 2),
                        'percent': usage.percent
                    })
                except PermissionError:
                    continue
            
            # Get I/O statistics
            io_counters = psutil.disk_io_counters(perdisk=True)
            if io_counters:
                for disk, counters in io_counters.items():
                    disk_info['io_stats'][disk] = {
                        'read_count': counters.read_count,
                        'write_count': counters.write_count,
                        'read_bytes': counters.read_bytes,
                        'write_bytes': counters.write_bytes
                    }
            
            return disk_info
        except Exception as e:
            return {'error': str(e)}

    def get_systemd_services(self) -> Dict[str, Any]:
        """
        Collect information about systemd services and their resource consumption.
        
        Returns:
            Dictionary containing systemd service information
        """
        try:
            services_info = {
                'enabled_services': [],
                'failed_services': [],
                'running_services': []
            }
            
            # Get list of enabled services
            try:
                result = subprocess.run(
                    ['systemctl', 'list-unit-files', '--type=service', '--state=enabled', '--no-pager'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n')[1:-2]:
                        if line.strip():
                            parts = line.split()
                            if parts:
                                services_info['enabled_services'].append(parts[0])
            except Exception:
                pass
            
            # Get list of failed services
            try:
                result = subprocess.run(
                    ['systemctl', 'list-units', '--type=service', '--state=failed', '--no-pager'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n')[1:-2]:
                        if line.strip() and 'loaded' in line:
                            parts = line.split()
                            if parts:
                                services_info['failed_services'].append(parts[0])
            except Exception:
                pass
            
            # Get list of running services
            try:
                result = subprocess.run(
                    ['systemctl', 'list-units', '--type=service', '--state=running', '--no-pager'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n')[1:-2]:
                        if line.strip() and 'loaded' in line:
                            parts = line.split()
                            if parts:
                                services_info['running_services'].append(parts[0])
            except Exception:
                pass
            
            return services_info
        except Exception as e:
            return {'error': str(e)}

    def get_package_info(self) -> Dict[str, Any]:
        """
        Collect package information (installed, orphaned packages).
        
        Returns:
            Dictionary containing package information
        """
        try:
            pkg_info = {
                'total_packages': 0,
                'explicit_packages': 0,
                'orphaned_packages': [],
                'package_managers': []
            }
            
            # Check if pacman is available (Arch Linux package manager)
            try:
                # Get total packages
                result = subprocess.run(
                    ['pacman', '-Q'],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    pkg_info['total_packages'] = len(result.stdout.strip().split('\n'))
                    pkg_info['package_managers'].append('pacman')
                
                # Get explicitly installed packages
                result = subprocess.run(
                    ['pacman', '-Qe'],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    pkg_info['explicit_packages'] = len(result.stdout.strip().split('\n'))
                
                # Get orphaned packages
                result = subprocess.run(
                    ['pacman', '-Qdtq'],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0 and result.stdout.strip():
                    pkg_info['orphaned_packages'] = result.stdout.strip().split('\n')
            except FileNotFoundError:
                pkg_info['error'] = 'pacman not found - this may not be an Arch Linux system'
            except Exception as e:
                pkg_info['error'] = f'Error collecting package info: {str(e)}'
            
            return pkg_info
        except Exception as e:
            return {'error': str(e)}

    def get_boot_info(self) -> Dict[str, Any]:
        """
        Collect boot time and systemd-analyze information.
        
        Returns:
            Dictionary containing boot information
        """
        try:
            boot_info = {
                'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat(),
                'uptime_seconds': int(psutil.time.time() - psutil.boot_time())
            }
            
            # Get systemd-analyze information
            try:
                result = subprocess.run(
                    ['systemd-analyze', 'time'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    boot_info['systemd_analyze'] = result.stdout.strip()
                
                # Get blame information (top 10 slowest services)
                result = subprocess.run(
                    ['systemd-analyze', 'blame'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')[:10]
                    boot_info['slowest_services'] = lines
            except Exception:
                pass
            
            return boot_info
        except Exception as e:
            return {'error': str(e)}

    def get_network_info(self) -> Dict[str, Any]:
        """
        Collect network configuration and usage.
        
        Returns:
            Dictionary containing network information
        """
        try:
            net_info = {
                'interfaces': {},
                'connections': 0
            }
            
            # Get network interfaces and their stats
            for interface, addrs in psutil.net_if_addrs().items():
                net_info['interfaces'][interface] = {
                    'addresses': [{'family': addr.family.name, 'address': addr.address} 
                                  for addr in addrs]
                }
            
            # Get network I/O statistics
            io_counters = psutil.net_io_counters(pernic=True)
            for interface, counters in io_counters.items():
                if interface in net_info['interfaces']:
                    net_info['interfaces'][interface]['stats'] = {
                        'bytes_sent': counters.bytes_sent,
                        'bytes_recv': counters.bytes_recv,
                        'packets_sent': counters.packets_sent,
                        'packets_recv': counters.packets_recv
                    }
            
            # Get number of network connections
            net_info['connections'] = len(psutil.net_connections())
            
            return net_info
        except Exception as e:
            return {'error': str(e)}

    def get_kernel_info(self) -> Dict[str, Any]:
        """
        Collect kernel version and loaded modules information.
        
        Returns:
            Dictionary containing kernel information
        """
        try:
            kernel_info = {}
            
            # Get kernel version
            try:
                result = subprocess.run(
                    ['uname', '-r'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    kernel_info['version'] = result.stdout.strip()
            except Exception:
                pass
            
            # Get kernel architecture
            try:
                result = subprocess.run(
                    ['uname', '-m'],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    kernel_info['architecture'] = result.stdout.strip()
            except Exception:
                pass
            
            # Get loaded kernel modules count
            try:
                result = subprocess.run(
                    ['lsmod'],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    # -1 to exclude header line
                    kernel_info['loaded_modules_count'] = len(result.stdout.strip().split('\n')) - 1
            except Exception:
                pass
            
            return kernel_info
        except Exception as e:
            return {'error': str(e)}

    def get_formatted_summary(self) -> str:
        """
        Get a human-readable summary of system information.
        
        Returns:
            Formatted string summary
        """
        if not self.info:
            self.collect_all()
        
        summary = []
        summary.append("=== Arch Linux System Information Summary ===\n")
        
        # CPU
        if 'cpu' in self.info and 'error' not in self.info['cpu']:
            cpu = self.info['cpu']
            summary.append(f"CPU Model: {cpu.get('model', 'Unknown')}")
            summary.append(f"CPU Cores: {cpu.get('count_physical', 'N/A')} physical, {cpu.get('count_logical', 'N/A')} logical")
            summary.append(f"CPU Usage: {cpu.get('usage_average', 0):.1f}%")
            summary.append(f"Load Average: {cpu.get('load_average', 'N/A')}\n")
        
        # Memory
        if 'memory' in self.info and 'error' not in self.info['memory']:
            mem = self.info['memory']
            if 'ram' in mem:
                summary.append(f"RAM: {mem['ram']['used_gb']:.2f}GB / {mem['ram']['total_gb']:.2f}GB ({mem['ram']['percent']:.1f}% used)")
            if 'swap' in mem:
                summary.append(f"Swap: {mem['swap']['used_gb']:.2f}GB / {mem['swap']['total_gb']:.2f}GB ({mem['swap']['percent']:.1f}% used)\n")
        
        # Disk
        if 'disk' in self.info and 'error' not in self.info['disk']:
            summary.append("Disk Usage:")
            for partition in self.info['disk'].get('partitions', []):
                summary.append(f"  {partition['mountpoint']}: {partition['used_gb']:.2f}GB / {partition['total_gb']:.2f}GB ({partition['percent']:.1f}% used)")
            summary.append("")
        
        # Services
        if 'services' in self.info and 'error' not in self.info['services']:
            svc = self.info['services']
            summary.append(f"Services: {len(svc.get('running_services', []))} running, {len(svc.get('enabled_services', []))} enabled")
            if svc.get('failed_services'):
                summary.append(f"Failed Services: {len(svc['failed_services'])}")
            summary.append("")
        
        # Packages
        if 'packages' in self.info and 'error' not in self.info['packages']:
            pkg = self.info['packages']
            summary.append(f"Packages: {pkg.get('total_packages', 0)} total, {pkg.get('explicit_packages', 0)} explicit")
            if pkg.get('orphaned_packages'):
                summary.append(f"Orphaned Packages: {len(pkg['orphaned_packages'])}")
            summary.append("")
        
        # Boot
        if 'boot' in self.info and 'error' not in self.info['boot']:
            boot = self.info['boot']
            uptime_hours = boot.get('uptime_seconds', 0) // 3600
            summary.append(f"Uptime: {uptime_hours} hours")
            if 'systemd_analyze' in boot:
                summary.append(f"Boot Time: {boot['systemd_analyze']}")
            summary.append("")
        
        # Kernel
        if 'kernel' in self.info and 'error' not in self.info['kernel']:
            kernel = self.info['kernel']
            summary.append(f"Kernel: {kernel.get('version', 'Unknown')} ({kernel.get('architecture', 'Unknown')})")
            summary.append(f"Loaded Modules: {kernel.get('loaded_modules_count', 'Unknown')}\n")
        
        return '\n'.join(summary)


if __name__ == '__main__':
    # Test the collector
    collector = SystemInfoCollector()
    info = collector.collect_all()
    print(collector.get_formatted_summary())
    print("\n=== Full JSON Output ===")
    print(json.dumps(info, indent=2, default=str))
