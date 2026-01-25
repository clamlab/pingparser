# Define variables for paths
$baseDir = "D:\clamlab\pingparser"
$venvDir = "$baseDir\.venv\Scripts"
$pythonScript = "$baseDir\src\dailyautoproc\autoproc_stick_v01.py"
$configPath   = "$baseDir\src\dailyautoproc\configs\config_v10.yaml"

# Change to the base directory
cd $baseDir

# Activate the virtual environment
& "$venvDir\Activate.ps1"

# Set PYTHONPATH to include the parent directory of pingparser
$env:PYTHONPATH = $baseDir

# Run the Python script with the desired arguments
python "$pythonScript" --config "$configPath"

Start-Sleep -Seconds 10 #pause before closing