# Define variables for paths
$baseDir = "D:\clamlab"
$venvDir = "$baseDir\.venv\Scripts"
$pythonScript = "$baseDir\pingparser\dailyautoproc\autoproc.py"
$configPath = "$baseDir\pingparser\dailyautoproc\configs\config_v06.yaml"

# Change to the base directory
cd $baseDir

# Activate the virtual environment
& "$venvDir\Activate.ps1"

# Set PYTHONPATH to include the parent directory of pingparser
$env:PYTHONPATH = $baseDir

# Run the Python script with the desired arguments
python "$pythonScript" --config "$configPath"
