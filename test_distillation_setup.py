#!/usr/bin/env python3
"""
Test script to verify the distillation setup.
This script checks if all components are properly configured.
"""

import os
import sys
from pathlib import Path

def check_file_exists(file_path, description):
    """Check if a file exists and print status."""
    if Path(file_path).exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (MISSING)")
        return False

def main():
    print("🔍 Testing ProtoMotions Distillation Setup\n")
    
    # Check agent files
    print("📁 Agent Files:")
    agent_files = [
        ("protomotions/agents/distillation/agent.py", "Distillation Agent"),
        ("protomotions/agents/distillation/model.py", "Distillation Model"),
    ]
    
    agent_ok = True
    for file_path, description in agent_files:
        if not check_file_exists(file_path, description):
            agent_ok = False
    
    # Check configuration files
    print("\n📁 Configuration Files:")
    config_files = [
        ("protomotions/config/agent/distillation/agent.yaml", "Distillation Agent Config"),
        ("protomotions/config/exp/distillation/unified_lightweight.yaml", "Unified Lightweight Config"),
        ("protomotions/config/opt/distillation/tasks/unified_evaluation.yaml", "Unified Evaluation Config"),
    ]
    
    config_ok = True
    for file_path, description in config_files:
        if not check_file_exists(file_path, description):
            config_ok = False
    
    # Check MaskedMimic configurations
    print("\n📁 MaskedMimic Configurations:")
    mimic_files = [
        ("protomotions/config/exp/masked_mimic/path_follower.yaml", "Path Follower Config"),
        ("protomotions/config/exp/masked_mimic/goal_directed.yaml", "Goal Directed Config"),
        ("protomotions/config/exp/masked_mimic/steering.yaml", "Steering Config"),
    ]
    
    mimic_ok = True
    for file_path, description in mimic_files:
        if not check_file_exists(file_path, description):
            mimic_ok = False
    
    # Check evaluation configurations
    print("\n📁 Evaluation Configurations:")
    eval_files = [
        ("protomotions/config/opt/masked_mimic/tasks/path_following.yaml", "Path Following Eval"),
        ("protomotions/config/opt/masked_mimic/tasks/goal_directed.yaml", "Goal Directed Eval"),
        ("protomotions/config/opt/masked_mimic/tasks/steering.yaml", "Steering Eval"),
    ]
    
    eval_ok = True
    for file_path, description in eval_files:
        if not check_file_exists(file_path, description):
            eval_ok = False
    
    # Check documentation
    print("\n📁 Documentation:")
    doc_files = [
        ("DISTILLATION_TRAINING_GUIDE.md", "Distillation Training Guide"),
        ("PATH_FOLLOWING_EVAL_GUIDE.md", "Path Following Evaluation Guide"),
    ]
    
    doc_ok = True
    for file_path, description in doc_files:
        if not check_file_exists(file_path, description):
            doc_ok = False
    
    # Summary
    print("\n" + "="*50)
    print("📊 SUMMARY")
    print("="*50)
    
    if all([agent_ok, config_ok, mimic_ok, eval_ok, doc_ok]):
        print("🎉 All components are properly set up!")
        print("\n🚀 You can now run the 4-step training pipeline:")
        print("   1. Full Body Tracker")
        print("   2. MaskedMimic")
        print("   3. Specialized Teachers")
        print("   4. Teacher-Student Distillation")
        print("\n📖 See DISTILLATION_TRAINING_GUIDE.md for detailed instructions.")
    else:
        print("⚠️  Some components are missing. Please check the files above.")
        print("\n🔧 Missing files need to be created before training.")
    
    print("\n" + "="*50)

if __name__ == "__main__":
    main() 