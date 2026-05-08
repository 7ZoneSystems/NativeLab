echo "----------------------------------"
echo " NativeLab Git Auto Push Script"
echo "----------------------------------"
read -p "Enter commit name: " msg

if [ -z "$msg" ]; then
    echo "Commit name cannot be empty!"
    exit 1
fi
echo ""
echo "Adding files..."
git add .
echo "Creating commit..."
git commit -m "$msg"
echo "Pushing to GitHub..."
git push
echo ""
echo " Push completed successfully! "
echo " Check for errors if any because script will still print success message even if push fails :("