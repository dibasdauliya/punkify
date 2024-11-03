import subprocess

# Run a command and capture the output
command = "git init"  # Replace with your command
result = subprocess.run(command, shell=True, capture_output=True, text=True)

# Print the command's output
print("Output:", result.stdout)
print("Error:", result.stderr)
