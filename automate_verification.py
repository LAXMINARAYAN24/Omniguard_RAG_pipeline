#!/usr/bin/env python
"""
automate_verification.py — Automated verification pipeline for OmniGuard-RAG

Runs the complete test suite, tracks changes with git, and generates
comprehensive reports. Designed for unattended execution with full monitoring.

Usage:
    python automate_verification.py                    # Full pipeline
    python automate_verification.py --quick            # Quick mode (3 seeds)
    python automate_verification.py --skip-baseline    # Skip single-seed baseline
    python automate_verification.py --skip-gwcc        # Skip GWCC diagnostic
    python automate_verification.py --skip-multiseed   # Skip multi-seed eval
"""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


class Colors:
    """ANSI color codes for terminal output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_section(title: str):
    """Print formatted section header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{title.center(80)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'=' * 80}{Colors.ENDC}\n")


def print_success(message: str):
    """Print success message"""
    print(f"{Colors.OKGREEN}✓ {message}{Colors.ENDC}")


def print_error(message: str):
    """Print error message"""
    print(f"{Colors.FAIL}✗ {message}{Colors.ENDC}")


def print_info(message: str):
    """Print info message"""
    print(f"{Colors.OKBLUE}ℹ {message}{Colors.ENDC}")


def print_warning(message: str):
    """Print warning message"""
    print(f"{Colors.WARNING}⚠ {message}{Colors.ENDC}")


def run_command(cmd: List[str], description: str, timeout: int = 600) -> Tuple[bool, str, float]:
    """
    Run a shell command and return success status, output, and elapsed time.

    Args:
        cmd: Command to execute as list of strings
        description: Human-readable description for logging
        timeout: Timeout in seconds (default: 10 minutes)

    Returns:
        (success, output, elapsed_time)
    """
    print_info(f"Running: {description}")
    print(f"  Command: {' '.join(cmd)}")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding='utf-8',
            errors='replace'
        )
        elapsed = time.time() - start_time

        if result.returncode == 0:
            print_success(f"Completed in {elapsed:.1f}s")
            return True, result.stdout, elapsed
        else:
            print_error(f"Failed with exit code {result.returncode}")
            print(f"  STDERR: {result.stderr[:500]}")
            return False, result.stderr, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print_error(f"Timeout after {elapsed:.1f}s")
        return False, f"Timeout after {timeout}s", elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print_error(f"Exception: {e}")
        return False, str(e), elapsed


def git_commit(message: str, files: List[str] = None) -> bool:
    """
    Commit changes to git with the given message.

    Args:
        message: Commit message
        files: Optional list of specific files to add; if None, adds all changes

    Returns:
        True if commit succeeded, False otherwise
    """
    # Add files
    if files:
        for f in files:
            success, output, _ = run_command(['git', 'add', f], f"Adding {f}", timeout=30)
            if not success:
                print_warning(f"Failed to add {f}, continuing anyway")
    else:
        success, output, _ = run_command(['git', 'add', '-A'], "Adding all changes", timeout=30)
        if not success:
            print_error("Failed to stage changes")
            return False

    # Check if there's anything to commit
    success, output, _ = run_command(['git', 'diff', '--cached', '--quiet'],
                                     "Checking for staged changes", timeout=30)
    if success:  # No changes (exit code 0 means no diff)
        print_info("No changes to commit")
        return True

    # Commit
    commit_msg = f"{message}\n\nCo-Authored-By: Claude Code <noreply@anthropic.com>"
    success, output, _ = run_command(['git', 'commit', '-m', commit_msg],
                                     "Committing changes", timeout=60)
    return success


def extract_metrics_from_output(output: str) -> Dict:
    """Extract key metrics from benchmark output for summary reporting"""
    metrics = {}
    lines = output.split('\n')

    for line in lines:
        # Look for OmniGuard-RAG results line
        if 'OmniGuard-RAG' in line and '|' in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 7:
                try:
                    metrics['omniguard'] = {
                        'accuracy': parts[1].rstrip('%'),
                        'overall_asr': parts[2].rstrip('%'),
                        'pidp_asr': parts[3].rstrip('%'),
                        'collusion_asr': parts[4].rstrip('%'),
                        'stealth_asr': parts[5].rstrip('%'),
                        'silent_asr': parts[6].rstrip('%'),
                    }
                except (IndexError, ValueError):
                    pass

        # Look for DRS FPR
        if 'held-out false-positive rate' in line and '%' in line:
            try:
                # Extract percentage value
                parts = line.split(':')
                if len(parts) >= 2:
                    fpr_str = parts[1].split('%')[0].strip()
                    metrics['drs_fpr'] = fpr_str + '%'
            except (IndexError, ValueError):
                pass

    return metrics


def run_pipeline(args) -> Dict:
    """
    Execute the complete verification pipeline.

    Returns:
        Dictionary containing results and timing information
    """
    results = {
        'start_time': datetime.now(timezone.utc).isoformat(),
        'stages': {},
        'success': True,
        'errors': []
    }

    # Stage 1: Baseline single-seed benchmark
    if not args.skip_baseline:
        print_section("STAGE 1: Baseline Single-Seed Benchmark")
        success, output, elapsed = run_command(
            ['python', 'run_omniguard_benchmark.py'],
            "Single-seed benchmark (seed=7, n=200)",
            timeout=300
        )

        results['stages']['baseline'] = {
            'success': success,
            'elapsed_seconds': elapsed,
            'metrics': extract_metrics_from_output(output) if success else {}
        }

        if success:
            git_commit("Automated baseline benchmark run completed",
                      ['results/', '*.log'])
        else:
            results['success'] = False
            results['errors'].append("Baseline benchmark failed")

    # Stage 2: GWCC diagnostic verification
    if not args.skip_gwcc:
        print_section("STAGE 2: GWCC Diagnostic Verification")
        success, output, elapsed = run_command(
            ['python', 'run_gwcc_diagnostic.py'],
            "GWCC mechanism verification",
            timeout=300
        )

        results['stages']['gwcc'] = {
            'success': success,
            'elapsed_seconds': elapsed,
        }

        if success:
            git_commit("Automated GWCC diagnostic run completed",
                      ['results/gwcc_diagnostic.md'])
        else:
            results['success'] = False
            results['errors'].append("GWCC diagnostic failed")

    # Stage 3: Multi-seed statistical evaluation
    if not args.skip_multiseed:
        print_section("STAGE 3: Multi-Seed Statistical Evaluation")

        cmd = ['python', 'run_full_evaluation.py']
        if args.quick:
            cmd.append('--quick')
            description = "Multi-seed evaluation (quick mode: 3 seeds × 60 queries)"
            timeout = 600
        else:
            description = "Multi-seed evaluation (8 seeds × 200 queries)"
            timeout = 1800

        success, output, elapsed = run_command(cmd, description, timeout=timeout)

        results['stages']['multiseed'] = {
            'success': success,
            'elapsed_seconds': elapsed,
            'metrics': extract_metrics_from_output(output) if success else {}
        }

        if success:
            git_commit("Automated multi-seed evaluation completed",
                      ['results/path_a_report.md', 'results/path_a_raw_results.json'])
        else:
            results['success'] = False
            results['errors'].append("Multi-seed evaluation failed")

    results['end_time'] = datetime.now(timezone.utc).isoformat()
    results['total_elapsed_seconds'] = sum(
        stage.get('elapsed_seconds', 0)
        for stage in results['stages'].values()
    )

    return results


def generate_summary_report(results: Dict, output_path: Path):
    """Generate a summary report of the pipeline execution"""
    lines = []
    lines.append("# OmniGuard-RAG Automated Verification Summary")
    lines.append("")
    lines.append(f"**Pipeline Started:** {results['start_time']}")
    lines.append(f"**Pipeline Completed:** {results['end_time']}")
    lines.append(f"**Total Duration:** {results['total_elapsed_seconds']:.1f} seconds "
                 f"({results['total_elapsed_seconds']/60:.1f} minutes)")
    lines.append(f"**Overall Status:** {'✓ SUCCESS' if results['success'] else '✗ FAILED'}")
    lines.append("")

    if results['errors']:
        lines.append("## Errors")
        lines.append("")
        for error in results['errors']:
            lines.append(f"- {error}")
        lines.append("")

    lines.append("## Stage Results")
    lines.append("")

    for stage_name, stage_data in results['stages'].items():
        status = "✓ PASS" if stage_data['success'] else "✗ FAIL"
        lines.append(f"### {stage_name.upper()} — {status}")
        lines.append(f"**Duration:** {stage_data['elapsed_seconds']:.1f}s")
        lines.append("")

        if 'metrics' in stage_data and stage_data['metrics']:
            metrics = stage_data['metrics']
            if 'omniguard' in metrics:
                lines.append("**OmniGuard-RAG Metrics:**")
                for k, v in metrics['omniguard'].items():
                    lines.append(f"- {k}: {v}")
                lines.append("")
            if 'drs_fpr' in metrics:
                lines.append(f"**DRS FPR:** {metrics['drs_fpr']}")
                lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*This report was generated automatically by automate_verification.py*")

    output_path.write_text('\n'.join(lines), encoding='utf-8')
    print_success(f"Summary report written to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: 3 seeds instead of 8, 60 queries instead of 200')
    parser.add_argument('--skip-baseline', action='store_true',
                       help='Skip baseline single-seed benchmark')
    parser.add_argument('--skip-gwcc', action='store_true',
                       help='Skip GWCC diagnostic verification')
    parser.add_argument('--skip-multiseed', action='store_true',
                       help='Skip multi-seed evaluation')
    parser.add_argument('--output-dir', type=Path, default=Path('results'),
                       help='Output directory for reports (default: results/)')

    args = parser.parse_args()

    print_section("OmniGuard-RAG Automated Verification Pipeline")
    print(f"Started: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"Mode: {'Quick (3 seeds, 60 queries)' if args.quick else 'Full (8 seeds, 200 queries)'}")
    print("")

    # Verify we're in a git repository
    success, output, _ = run_command(['git', 'status'], "Checking git repository", timeout=10)
    if not success:
        print_error("Not in a git repository or git not available")
        print_info("Initialize git with: git init")
        sys.exit(1)

    # Run the pipeline
    start_total = time.time()
    results = run_pipeline(args)
    total_elapsed = time.time() - start_total

    # Generate summary report
    args.output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    summary_path = args.output_dir / f'automation_summary_{timestamp}.md'
    generate_summary_report(results, summary_path)

    # Save detailed JSON results
    json_path = args.output_dir / f'automation_results_{timestamp}.json'
    json_path.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print_success(f"Detailed results written to {json_path}")

    # Final commit with summary
    git_commit(f"Automated verification pipeline completed: {timestamp}",
               [str(summary_path), str(json_path)])

    # Print final summary
    print_section("PIPELINE COMPLETE")
    print(f"Total elapsed time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"Overall status: {'✓ SUCCESS' if results['success'] else '✗ FAILED'}")

    if results['success']:
        print("")
        print_success("All verification stages passed!")
        print_info("Review results in:")
        print(f"  - {summary_path}")
        print(f"  - results/path_a_report.md")
        print(f"  - results/gwcc_diagnostic.md")
        sys.exit(0)
    else:
        print("")
        print_error("Some verification stages failed")
        print_info("Check the summary report for details:")
        print(f"  - {summary_path}")
        sys.exit(1)


if __name__ == '__main__':
    main()
