#!/bin/bash
# ─────────────────────────────────────────────────────
# Biotech IPO Dashboard — One-click deploy script
# Run this from the biotech-ipo-dashboard folder
# ─────────────────────────────────────────────────────

set -e

echo ""
echo "🧬 Biotech IPO Dashboard — Deploy"
echo "──────────────────────────────────"
echo ""

# Check for gh CLI
if ! command -v gh &> /dev/null; then
    echo "⚠️  GitHub CLI (gh) not found. Installing..."
    if command -v brew &> /dev/null; then
        brew install gh
    elif command -v winget &> /dev/null; then
        winget install --id GitHub.cli
    else
        echo "Please install GitHub CLI first: https://cli.github.com"
        exit 1
    fi
fi

# Check auth
if ! gh auth status &> /dev/null 2>&1; then
    echo "📋 You need to log into GitHub first."
    echo ""
    gh auth login
fi

echo ""
echo "📦 Step 1/3: Creating GitHub repo..."
REPO_NAME="biotech-ipo-dashboard"

# Init git and push
git init -q
git add -A
git commit -q -m "Initial commit — Biotech IPO Dashboard"

# Create the repo (private by default)
gh repo create "$REPO_NAME" --private --source=. --push

echo "✅ Repo created and pushed!"
echo ""

# Step 3: Open Streamlit Cloud
echo "🚀 Step 2/3: Opening Streamlit Cloud..."
echo ""
echo "   I'll open Streamlit Cloud in your browser."
echo "   Once there:"
echo "   1. Click 'New app'"
echo "   2. Select repo: $REPO_NAME"
echo "   3. Branch: main"
echo "   4. Main file: app.py"
echo "   5. Click 'Deploy'"
echo ""

# Open Streamlit Cloud
if command -v open &> /dev/null; then
    open "https://share.streamlit.io"
elif command -v xdg-open &> /dev/null; then
    xdg-open "https://share.streamlit.io"
elif command -v start &> /dev/null; then
    start "https://share.streamlit.io"
else
    echo "   → Go to: https://share.streamlit.io"
fi

echo "──────────────────────────────────"
echo "Step 3/3: Once deployed, share the URL with your team!"
echo ""
echo "   To add new IPOs later, just edit app.py,"
echo "   push to GitHub, and the dashboard auto-updates."
echo "──────────────────────────────────"
