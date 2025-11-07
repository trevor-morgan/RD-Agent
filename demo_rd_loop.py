#!/usr/bin/env python3
"""
Real-World RD-Agent Demo: Feature Engineering for Machine Learning
Demonstrates the complete Research → Development loop without Docker

Scenario: Automatically propose and test feature engineering ideas for a dataset
"""

import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent))

from rdagent.utils.env import LocalConf, LocalEnv

def create_sample_dataset():
    """Create a sample dataset for demonstration"""
    workspace = Path("/tmp/rdagent_ml_demo")
    workspace.mkdir(exist_ok=True)

    # Create a simple dataset
    dataset_code = '''
import pandas as pd
import numpy as np

# Generate sample data
np.random.seed(42)
n_samples = 1000

data = pd.DataFrame({
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.randint(20000, 150000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'loan_amount': np.random.randint(5000, 50000, n_samples),
})

# Target: loan default (synthetic)
data['default'] = ((data['income'] < 40000) &
                   (data['credit_score'] < 600) &
                   (data['loan_amount'] > 30000)).astype(int)

# Save dataset
data.to_csv('loan_data.csv', index=False)
print(f"Dataset created: {len(data)} samples")
print(f"Features: {list(data.columns)}")
print(f"Default rate: {data['default'].mean():.2%}")
'''

    (workspace / "create_dataset.py").write_text(dataset_code)
    return workspace

def run_research_phase(env, workspace):
    """Research Phase: Propose feature engineering ideas"""
    print("\n" + "="*70)
    print("🔬 RESEARCH PHASE: Proposing Feature Ideas")
    print("="*70)

    # In real RD-Agent, this would use LLM to propose ideas
    # Here we simulate with predefined ideas
    ideas = [
        {
            "name": "debt_to_income_ratio",
            "description": "Ratio of loan amount to annual income",
            "formula": "loan_amount / income"
        },
        {
            "name": "credit_loan_interaction",
            "description": "Interaction between credit score and loan amount",
            "formula": "credit_score * loan_amount / 1000000"
        },
        {
            "name": "age_income_normalized",
            "description": "Income normalized by age",
            "formula": "income / age"
        }
    ]

    print("\n💡 Proposed Features:")
    for i, idea in enumerate(ideas, 1):
        print(f"\n{i}. {idea['name']}")
        print(f"   Description: {idea['description']}")
        print(f"   Formula: {idea['formula']}")

    return ideas

def run_development_phase(env, workspace, ideas):
    """Development Phase: Implement the features"""
    print("\n" + "="*70)
    print("💻 DEVELOPMENT PHASE: Implementing Features")
    print("="*70)

    # Generate implementation code
    impl_code = '''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

# Load data
data = pd.read_csv('loan_data.csv')
print("Original features:", list(data.columns))

# Implement proposed features
'''

    for idea in ideas:
        impl_code += f"\n# Feature: {idea['name']}\n"
        # Replace column names with data['column'] references
        formula = idea['formula']
        for col in ['loan_amount', 'income', 'credit_score', 'age']:
            formula = formula.replace(col, f"data['{col}']")
        impl_code += f"data['{idea['name']}'] = {formula}\n"

    impl_code += '''
# Prepare data
X = data.drop('default', axis=1)
y = data['default']

print(f"\\nAll features: {list(X.columns)}")
print(f"Dataset shape: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train baseline model (original features only)
original_features = ['age', 'income', 'credit_score', 'loan_amount']
clf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
clf_baseline.fit(X_train[original_features], y_train)
y_pred_baseline = clf_baseline.predict(X_test[original_features])

baseline_acc = accuracy_score(y_test, y_pred_baseline)
baseline_auc = roc_auc_score(y_test, clf_baseline.predict_proba(X_test[original_features])[:, 1])

print(f"\\n📊 Baseline Model (Original Features):")
print(f"   Accuracy: {baseline_acc:.4f}")
print(f"   ROC-AUC: {baseline_auc:.4f}")

# Train enhanced model (with new features)
clf_enhanced = RandomForestClassifier(n_estimators=100, random_state=42)
clf_enhanced.fit(X_train, y_train)
y_pred_enhanced = clf_enhanced.predict(X_test)

enhanced_acc = accuracy_score(y_test, y_pred_enhanced)
enhanced_auc = roc_auc_score(y_test, clf_enhanced.predict_proba(X_test)[:, 1])

print(f"\\n✨ Enhanced Model (With New Features):")
print(f"   Accuracy: {enhanced_acc:.4f}")
print(f"   ROC-AUC: {enhanced_auc:.4f}")

# Calculate improvement
acc_improvement = (enhanced_acc - baseline_acc) / baseline_acc * 100
auc_improvement = (enhanced_auc - baseline_auc) / baseline_auc * 100

print(f"\\n📈 Improvement:")
print(f"   Accuracy: {acc_improvement:+.2f}%")
print(f"   ROC-AUC: {auc_improvement:+.2f}%")

# Feature importance
importances = clf_enhanced.feature_importances_
feature_importance = sorted(
    zip(X.columns, importances),
    key=lambda x: x[1],
    reverse=True
)

print(f"\\n🎯 Top 5 Most Important Features:")
for i, (feat, imp) in enumerate(feature_importance[:5], 1):
    marker = "🆕" if feat not in original_features else "📌"
    print(f"   {i}. {marker} {feat}: {imp:.4f}")

# Save results
results = {
    "baseline_accuracy": float(baseline_acc),
    "baseline_auc": float(baseline_auc),
    "enhanced_accuracy": float(enhanced_acc),
    "enhanced_auc": float(enhanced_auc),
    "accuracy_improvement": float(acc_improvement),
    "auc_improvement": float(auc_improvement),
    "top_features": [(f, float(i)) for f, i in feature_importance[:5]]
}

import json
with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\\n✅ Results saved to results.json")
'''

    (workspace / "implement_features.py").write_text(impl_code)

    print("\n📝 Generated implementation code")
    print(f"   File: implement_features.py")
    print(f"   Features to implement: {len(ideas)}")

    return "implement_features.py"

def run_execution_phase(env, workspace):
    """Execution Phase: Run the implementation"""
    print("\n" + "="*70)
    print("⚙️ EXECUTION PHASE: Running Experiments")
    print("="*70)

    # Use sys.executable to ensure subprocess uses same Python with all packages
    python_cmd = sys.executable

    print("\n▶️ Step 1: Creating dataset...")
    result = env.run(
        entry=f"{python_cmd} create_dataset.py",
        local_path=str(workspace)
    )
    print(result.stdout if hasattr(result, 'stdout') else str(result))

    print("\n▶️ Step 2: Implementing and testing features...")
    result = env.run(
        entry=f"{python_cmd} implement_features.py",
        local_path=str(workspace)
    )
    output = result.stdout if hasattr(result, 'stdout') else str(result)
    print(output)

    return output, workspace / "results.json"

def run_evaluation_phase(results_file):
    """Evaluation Phase: Analyze results and provide feedback"""
    print("\n" + "="*70)
    print("📊 EVALUATION PHASE: Analyzing Results")
    print("="*70)

    if results_file.exists():
        results = json.loads(results_file.read_text())

        print("\n📈 Performance Summary:")
        print(f"   Baseline Accuracy:  {results['baseline_accuracy']:.4f}")
        print(f"   Enhanced Accuracy:  {results['enhanced_accuracy']:.4f}")
        print(f"   Improvement:        {results['accuracy_improvement']:+.2f}%")

        print(f"\n   Baseline ROC-AUC:   {results['baseline_auc']:.4f}")
        print(f"   Enhanced ROC-AUC:   {results['enhanced_auc']:.4f}")
        print(f"   Improvement:        {results['auc_improvement']:+.2f}%")

        # Provide feedback
        print("\n💭 Feedback:")
        if results['accuracy_improvement'] > 5:
            print("   ✅ Excellent! Features significantly improved model performance")
            print("   → These features should be kept in production")
        elif results['accuracy_improvement'] > 0:
            print("   ✅ Good! Features provide marginal improvement")
            print("   → Consider keeping most valuable features")
        else:
            print("   ⚠️  Features did not improve performance")
            print("   → Try different feature engineering approaches")

        print("\n🔄 Next Iteration Ideas:")
        print("   • Try polynomial features")
        print("   • Explore binning/discretization")
        print("   • Test interaction terms with age")

        return results
    else:
        print("⚠️  Results file not found")
        return None

def demonstrate_rd_loop():
    """Main function demonstrating the complete R&D loop"""

    print("\n" + "🎯"*35)
    print("  RD-Agent: Real-World Feature Engineering Demo")
    print("  Research → Develop → Execute → Evaluate Loop")
    print("🎯"*35)

    print("\n📋 Scenario:")
    print("   Problem: Predict loan defaults")
    print("   Approach: Automated feature engineering")
    print("   Method: Propose features → Implement → Test → Analyze")

    # Setup
    workspace = create_sample_dataset()
    config = LocalConf(default_entry=sys.executable)
    env = LocalEnv(conf=config)

    print(f"\n🔧 Configuration:")
    print(f"   Environment: LocalEnv (no Docker)")
    print(f"   Workspace: {workspace}")
    print(f"   Python: {sys.version.split()[0]}")

    try:
        # Phase 1: Research - Propose ideas
        ideas = run_research_phase(env, workspace)

        # Phase 2: Development - Implement features
        impl_file = run_development_phase(env, workspace, ideas)

        # Phase 3: Execution - Run experiments
        output, results_file = run_execution_phase(env, workspace)

        # Phase 4: Evaluation - Analyze and provide feedback
        results = run_evaluation_phase(results_file)

        # Summary
        print("\n" + "="*70)
        print("🎉 R&D Loop Completed Successfully!")
        print("="*70)

        print("\n📊 What Happened:")
        print("   1. ✅ Research: Proposed 3 feature engineering ideas")
        print("   2. ✅ Development: Generated implementation code")
        print("   3. ✅ Execution: Ran experiments with 1000 samples")
        print("   4. ✅ Evaluation: Analyzed results and provided feedback")

        if results:
            print(f"\n🎯 Key Result:")
            print(f"   New features improved accuracy by {results['accuracy_improvement']:+.2f}%")
            print(f"   and ROC-AUC by {results['auc_improvement']:+.2f}%")

        print("\n💡 This Demonstrates:")
        print("   • Complete R&D workflow automation")
        print("   • LocalEnv execution (no Docker needed)")
        print("   • Iterative improvement cycle")
        print("   • Real machine learning evaluation")

        print("\n🔄 In Full RD-Agent:")
        print("   • LLM proposes features based on data analysis")
        print("   • Multiple iterations to refine features")
        print("   • Automatic code generation and debugging")
        print("   • Knowledge base for learning from past experiments")

        return True

    except Exception as e:
        print(f"\n❌ Error during R&D loop: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = demonstrate_rd_loop()
    sys.exit(0 if success else 1)
