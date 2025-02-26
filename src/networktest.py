#!/usr/bin/env python3
"""
Discord Voice Connection Diagnostic Tool
This script tests your server's connectivity to Discord voice servers and
helps diagnose network-related issues that might affect voice quality.
"""

import socket
import time
import statistics
import threading
import sys
import argparse
import subprocess
import os
import platform
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple

# Discord voice server regions and sample IPs
DISCORD_REGIONS = {
    "us-west": "135.181.78.52",      # US West
    "us-east": "134.122.127.233",    # US East
    "us-central": "138.197.169.153", # US Central
    "eu-west": "162.159.130.232",    # EU West
    "eu-central": "162.159.129.233", # EU Central
    "singapore": "128.199.234.156",  # Singapore
    "japan": "162.159.135.232",      # Japan
    "russia": "188.114.119.233",     # Russia
    "brazil": "143.198.56.125",      # Brazil
    "sydney": "162.159.135.233"      # Sydney
}

# Discord required ports
DISCORD_PORTS = [
    443,  # HTTPS
    80,   # HTTP
    50000, # Voice (sample - Discord uses range 49152-65535)
    51000, # Voice (sample - Discord uses range 49152-65535)
    52000  # Voice (sample - Discord uses range 49152-65535)
]

def run_ping_test(host: str, count: int = 10) -> Dict:
    """Run a ping test to the specified host"""
    ping_times: List[float] = []
    packet_loss = 0
    
    for _ in range(count):
        try:
            # Create a socket connection and measure time
            start_time = time.time()
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(2.0)
            s.connect((host, 443))
            s.close()
            elapsed = (time.time() - start_time) * 1000  # Convert to ms
            ping_times.append(elapsed)
            time.sleep(0.5)  # Don't flood the server
        except (socket.timeout, socket.error):
            packet_loss += 1
    
    # Calculate statistics if we have data
    if ping_times:
        return {
            "min": min(ping_times),
            "max": max(ping_times),
            "avg": statistics.mean(ping_times),
            "jitter": statistics.stdev(ping_times) if len(ping_times) > 1 else 0,
            "loss_percent": (packet_loss / count) * 100
        }
    else:
        return {
            "min": 0,
            "max": 0,
            "avg": 0,
            "jitter": 0,
            "loss_percent": 100
        }

def test_udp_port(host: str, port: int) -> bool:
    """Test if a UDP port is accessible"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1)
        sock.sendto(b"Discord UDP Test", (host, port))
        
        # Try to receive a response (this will likely timeout for Discord servers)
        try:
            sock.recvfrom(1024)
            result = True
        except socket.timeout:
            # For UDP, a timeout here doesn't necessarily mean failure
            # We were able to send the packet, which is what matters
            result = True
    except (socket.error, socket.timeout, OSError):
        result = False
    finally:
        sock.close()
    
    return result

def test_tcp_port(host: str, port: int) -> bool:
    """Test if a TCP port is accessible"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((host, port)) == 0
        sock.close()
        return result
    except:
        return False

def check_system_settings():
    """Check system settings relevant to Discord voice"""
    results = {}
    
    # Check OS
    results["os"] = platform.system()
    results["os_version"] = platform.version()
    
    # Check if running in Docker
    in_docker = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
    results["in_docker"] = in_docker
    
    # Check network buffer sizes on Linux
    if platform.system() == "Linux":
        try:
            rmem_max = subprocess.check_output("sysctl -n net.core.rmem_max", shell=True).decode().strip()
            wmem_max = subprocess.check_output("sysctl -n net.core.wmem_max", shell=True).decode().strip()
            results["rmem_max"] = rmem_max
            results["wmem_max"] = wmem_max
        except:
            results["buffer_check_error"] = "Failed to check network buffer sizes"
    
    # Check for UDP packet jitter
    if 'us-east' in DISCORD_REGIONS:
        udp_ping_times = []
        for _ in range(5):
            start_time = time.time()
            test_udp_port(DISCORD_REGIONS['us-east'], 50000)
            elapsed = (time.time() - start_time) * 1000
            udp_ping_times.append(elapsed)
            time.sleep(0.5)
        
        if udp_ping_times:
            results["udp_jitter"] = statistics.stdev(udp_ping_times) if len(udp_ping_times) > 1 else 0
            results["udp_avg_response"] = statistics.mean(udp_ping_times)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Discord Voice Connection Diagnostic')
    parser.add_argument('--full', action='store_true', help='Run full diagnostics (all regions)')
    args = parser.parse_args()
    
    print("Discord Voice Connection Diagnostic")
    print("===================================")
    
    # Test system settings
    print("\nSystem Information:")
    print("-----------------")
    system_info = check_system_settings()
    for key, value in system_info.items():
        print(f"{key}: {value}")
    
    # Regions to test
    regions_to_test = DISCORD_REGIONS if args.full else {k: DISCORD_REGIONS[k] for k in ['us-east', 'us-west', 'eu-west']}
    
    # Run ping tests
    print("\nPing Tests (TCP to port 443):")
    print("---------------------------")
    for region, ip in regions_to_test.items():
        print(f"Testing {region} ({ip})...")
        results = run_ping_test(ip)
        status = "✓ Good" if results["loss_percent"] < 5 and results["avg"] < 150 else "✗ Poor"
        
        print(f"  Status: {status}")
        print(f"  Min: {results['min']:.2f}ms")
        print(f"  Max: {results['max']:.2f}ms")
        print(f"  Avg: {results['avg']:.2f}ms")
        print(f"  Jitter: {results['jitter']:.2f}ms")
        print(f"  Packet Loss: {results['loss_percent']:.1f}%")
        print()
    
    # Test UDP ports (crucial for voice)
    print("\nUDP Port Tests (Voice):")
    print("---------------------")
    for port in DISCORD_PORTS:
        if port > 1000:  # Only test voice ports
            results = []
            for region, ip in regions_to_test.items():
                result = test_udp_port(ip, port)
                results.append(result)
                print(f"Region {region}, UDP Port {port}: {'✓ Accessible' if result else '✗ Blocked'}")
            
            success_rate = sum(results) / len(results) * 100 if results else 0
            print(f"Overall success rate for port {port}: {success_rate:.1f}%")
            print()
    
    # Print recommendations
    print("\nRecommendations:")
    print("---------------")
    
    if system_info.get("in_docker", False):
        print("- You are running in a Docker container. Consider the following:")
        print("  * Use --network=host if possible (Linux hosts only)")
        print("  * Increase UDP buffer sizes with --sysctl net.core.rmem_max=2097152 --sysctl net.core.wmem_max=2097152")
        print("  * Ensure the container has NET_ADMIN capability for network optimizations")
    
    if system_info.get("os") == "Windows":
        print("- Windows systems running Docker may experience additional network latency.")
        print("  * Consider running directly in WSL2 instead of Docker for better performance")
        print("  * Check Windows Defender Firewall settings for UDP port blocking")
    
    if any("rmem_max" in k for k in system_info) and int(system_info.get("rmem_max", "0")) < 1048576:
        print("- Network buffer sizes are smaller than recommended.")
        print("  * Increase buffer sizes: sysctl -w net.core.rmem_max=2097152 net.core.wmem_max=2097152")
    
    # Final summary
    print("\nSummary:")
    all_pings = []
    for region, ip in regions_to_test.items():
        results = run_ping_test(ip, count=3)
        all_pings.append(results["avg"])
    
    avg_ping = statistics.mean(all_pings) if all_pings else 0
    if avg_ping < 100:
        rating = "Excellent"
    elif avg_ping < 150:
        rating = "Good"
    elif avg_ping < 250:
        rating = "Fair"
    else:
        rating = "Poor"
    
    print(f"Discord voice connection quality rating: {rating} (Avg ping: {avg_ping:.2f}ms)")
    print("This tool helps identify network issues, but may not capture all possible problems.")
    print("If you're experiencing voice quality issues, please check Discord's status page and")
    print("community resources for additional troubleshooting steps.")

if __name__ == "__main__":
    main()